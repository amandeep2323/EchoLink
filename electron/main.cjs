const path = require("node:path");
const fs = require("node:fs");
const http = require("node:http");
const { spawn, spawnSync } = require("node:child_process");
const { app, BrowserWindow, shell, dialog } = require("electron");

const DEFAULT_DEV_SERVER_URL = "http://127.0.0.1:5173";
const BACKEND_HOST = "127.0.0.1";
const BACKEND_PORT = 8765;
const BACKEND_HEALTH_PATH = "/health";
const BACKEND_START_TIMEOUT_MS = 30000;
const BACKEND_RETRY_DELAY_MS = 1500;
const isDev = !app.isPackaged;

let mainWindow = null;
let backendProcess = null;
let isQuitting = false;
let logFilePath = null;
let backendStartInProgress = false;
let backendHealthMonitor = null;

function initLogging() {
  try {
    const logDir = path.join(app.getPath("userData"), "logs");
    fs.mkdirSync(logDir, { recursive: true });
    logFilePath = path.join(logDir, "main.log");
    appendLog("---- EchoLink session start ----");
  } catch (error) {
    console.warn(`[Electron] Failed to initialize logging: ${error.message}`);
  }
}

function appendLog(message) {
  if (!logFilePath) {
    return;
  }
  const timestamp = new Date().toISOString();
  try {
    fs.appendFileSync(logFilePath, `[${timestamp}] ${message}\n`, "utf8");
  } catch {
    // Ignore logging errors to avoid crashing the app.
  }
}

function log(message) {
  console.log(`[Electron] ${message}`);
  appendLog(`[Electron] ${message}`);
}

function getProjectRoot() {
  return app.getAppPath();
}

function findDevPythonExecutable(projectRoot) {
  const envPython = process.env.ECHOLINK_PYTHON;
  const candidates = [
    { command: envPython, args: [] },
    { command: path.join(projectRoot, ".venv", "Scripts", "python.exe"), args: [] },
    { command: path.join(projectRoot, "venv", "Scripts", "python.exe"), args: [] },
    { command: path.join(projectRoot, "python-backend", ".venv", "Scripts", "python.exe"), args: [] },
    { command: path.join(projectRoot, "python-backend", "venv", "Scripts", "python.exe"), args: [] },
    { command: "py", args: ["-3.12"] },
    { command: "py", args: ["-3.11"] },
    { command: "python", args: [] },
    { command: "py", args: ["-3"] },
  ].filter(Boolean);

  for (const candidate of candidates) {
    if (!candidate.command) {
      continue;
    }

    if (candidate.command === "python" || candidate.command === "py") {
      if (pythonWorks(candidate.command, candidate.args)) {
        return candidate;
      }
      continue;
    }

    if (fs.existsSync(candidate.command) && pythonWorks(candidate.command, candidate.args)) {
      return candidate;
    }
  }

  return null;
}

function pythonWorks(command, args = []) {
  const result = spawnSync(command, [...args, "-c", "import sys; print(sys.version)"], {
    windowsHide: true,
    stdio: "ignore",
  });
  return result.status === 0;
}

function resolveBackendCommand() {
  if (app.isPackaged) {
    const backendDir = path.join(process.resourcesPath, "backend");
    const backendExe = path.join(backendDir, "echolink-backend.exe");
    return {
      command: backendExe,
      args: [],
      cwd: backendDir,
    };
  }

  const projectRoot = getProjectRoot();
  const backendDir = path.join(projectRoot, "python-backend");
  const backendEntry = path.join(backendDir, "main.py");
  const python = findDevPythonExecutable(projectRoot);
  if (!python) {
    throw new Error(
      "No Python interpreter found. Set ECHOLINK_PYTHON or install Python/py launcher."
    );
  }

  return {
    command: python.command,
    args: [...python.args, backendEntry],
    cwd: backendDir,
  };
}

function attachBackendLogging(proc) {
  proc.on("error", (error) => {
    log(`Backend spawn error: ${error.message}`);
  });

  proc.stdout?.on("data", (chunk) => {
    process.stdout.write(`[Backend] ${chunk}`);
    appendLog(`[Backend] ${chunk.toString()}`.trimEnd());
  });

  proc.stderr?.on("data", (chunk) => {
    process.stderr.write(`[Backend] ${chunk}`);
    appendLog(`[Backend] ${chunk.toString()}`.trimEnd());
  });

  proc.once("exit", (code, signal) => {
    if (backendProcess === proc) {
      backendProcess = null;
    }
    log(`Backend exited (code=${code}, signal=${signal ?? "none"})`);

    if (!isQuitting) {
      log("Backend exited while app is still running.");
    }
  });
}

function spawnBackend(backend, options = {}) {
  const { useShell = false, windowsHide = true } = options;
  const proc = spawn(backend.command, backend.args, {
    cwd: backend.cwd,
    windowsHide,
    stdio: ["ignore", "pipe", "pipe"],
    shell: useShell,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1",
    },
  });

  attachBackendLogging(proc);
  return proc;
}

function checkBackendHealth(timeoutMs = 1500) {
  return new Promise((resolve) => {
    const req = http.request(
      {
        host: BACKEND_HOST,
        port: BACKEND_PORT,
        path: BACKEND_HEALTH_PATH,
        method: "GET",
        timeout: timeoutMs,
      },
      (res) => {
        res.resume();
        resolve(res.statusCode === 200);
      }
    );

    req.on("timeout", () => {
      req.destroy();
      resolve(false);
    });

    req.on("error", () => resolve(false));
    req.end();
  });
}

async function waitForBackendReady(maxWaitMs) {
  const startTime = Date.now();
  while (Date.now() - startTime < maxWaitMs) {
    const healthy = await checkBackendHealth();
    if (healthy) {
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return false;
}

function startHealthMonitor() {
  if (backendHealthMonitor) {
    return;
  }

  backendHealthMonitor = setInterval(async () => {
    const healthy = await checkBackendHealth();
    if (healthy) {
      log("Backend became healthy.");
      clearInterval(backendHealthMonitor);
      backendHealthMonitor = null;
    }
  }, 2000);
}

function stopHealthMonitor() {
  if (backendHealthMonitor) {
    clearInterval(backendHealthMonitor);
    backendHealthMonitor = null;
  }
}

function showBackendError(message) {
  log(message);
  if (!app.isReady()) {
    return;
  }
  try {
    dialog.showErrorBox("EchoLink Backend Failed to Start", message);
  } catch (error) {
    log(`Failed to show backend error dialog: ${error.message}`);
  }
}

async function startBackend() {
  if (backendProcess || backendStartInProgress) {
    return;
  }

  backendStartInProgress = true;
  const backend = resolveBackendCommand();

  if (!fs.existsSync(backend.cwd)) {
    showBackendError(`Backend working directory not found: ${backend.cwd}`);
    backendStartInProgress = false;
    return;
  }
  if (app.isPackaged && !fs.existsSync(backend.command)) {
    showBackendError(`Packaged backend executable not found: ${backend.command}`);
    backendStartInProgress = false;
    return;
  }

  const tryStart = async (label, options) => {
    log(`Starting backend (${label}): ${backend.command} ${backend.args.join(" ")}`);
    backendProcess = spawnBackend(backend, options);

    const healthy = await waitForBackendReady(BACKEND_START_TIMEOUT_MS);
    if (healthy) {
      log(`Backend healthy (${label}).`);
      stopHealthMonitor();
      return true;
    }

    if (backendProcess) {
      log(`Backend still starting (${label}) — continuing to monitor.`);
      startHealthMonitor();
      return true;
    }

    log(`Backend failed to become healthy (${label}).`);
    return false;
  };

  let started = await tryStart("direct", { useShell: false, windowsHide: true });

  if (!started && app.isPackaged) {
    await new Promise((resolve) => setTimeout(resolve, BACKEND_RETRY_DELAY_MS));
    started = await tryStart("shell", { useShell: true, windowsHide: false });
  }

  if (!started) {
    const logHint = logFilePath ? `See log: ${logFilePath}` : "Check logs for details.";
    showBackendError(`Backend failed to start. ${logHint}`);
  }

  backendStartInProgress = false;
}

function stopBackend() {
  return new Promise((resolve) => {
    if (!backendProcess) {
      stopHealthMonitor();
      resolve();
      return;
    }

    const proc = backendProcess;
    backendProcess = null;

    const finalize = () => resolve();
    proc.once("exit", finalize);

    try {
      if (process.platform === "win32") {
        const killer = spawn("taskkill", ["/pid", String(proc.pid), "/t", "/f"], {
          windowsHide: true,
          stdio: "ignore",
        });
        killer.once("exit", finalize);
      } else {
        proc.kill("SIGTERM");
        setTimeout(() => {
          if (!proc.killed) {
            proc.kill("SIGKILL");
          }
        }, 3000);
      }
    } catch (error) {
      log(`Error while stopping backend: ${error.message}`);
      resolve();
    }
  });
}

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 760,
    backgroundColor: "#0f172a",
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      webSecurity: true,
    },
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url).catch(() => {});
    return { action: "deny" };
  });

  mainWindow.webContents.on("will-navigate", (event, navigationUrl) => {
    const devUrl = process.env.VITE_DEV_SERVER_URL || DEFAULT_DEV_SERVER_URL;
    const allowedDev = isDev && navigationUrl.startsWith(devUrl);
    const allowedFile = navigationUrl.startsWith("file://");
    if (!allowedDev && !allowedFile) {
      event.preventDefault();
      shell.openExternal(navigationUrl).catch(() => {});
    }
  });

  if (isDev) {
    const devUrl = process.env.VITE_DEV_SERVER_URL || DEFAULT_DEV_SERVER_URL;
    mainWindow.loadURL(devUrl);
    mainWindow.webContents.openDevTools({ mode: "detach" });
  } else {
    mainWindow.loadFile(path.join(app.getAppPath(), "dist", "index.html"));
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

app.on("before-quit", (event) => {
  if (isQuitting) {
    return;
  }
  isQuitting = true;
  event.preventDefault();
  stopBackend().finally(() => {
    app.exit(0);
  });
});

app.whenReady().then(() => {
  initLogging();
  void startBackend();
  createMainWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createMainWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
