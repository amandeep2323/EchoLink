const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  platform: process.platform,
});

// AvatarLink mode/window control bridge (sandboxed).
contextBridge.exposeInMainWorld("avatarAPI", {
  switchMode: (mode) => ipcRenderer.invoke("mode:switch", mode), // 'echolink' | 'avatar'
  closeAvatar: () => ipcRenderer.invoke("avatar:close"),
  getMode: () => ipcRenderer.invoke("mode:get"),
  onModeChanged: (cb) =>
    ipcRenderer.on("mode:changed", (_e, mode) => cb(mode)),
});
