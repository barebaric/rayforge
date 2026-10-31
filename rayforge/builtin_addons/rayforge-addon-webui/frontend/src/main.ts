import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/layout.css";
import "./styles/components.css";
import { startApp } from "./App";

const root = document.getElementById("app");
if (root) {
  startApp(root);
}
