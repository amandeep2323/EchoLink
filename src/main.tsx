import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { App } from "./App";
import { AvatarApp } from "./avatar/AvatarApp";
import { getModeFromLocation } from "./avatar/modeRouter";

const mode = getModeFromLocation();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {mode === "avatar" ? <AvatarApp /> : <App />}
  </StrictMode>
);
