import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const source = join(root, "node_modules", "cesium", "Build", "Cesium");
const destination = join(root, "public", "cesium");
const folders = ["Workers", "ThirdParty", "Assets", "Widgets"];

if (!existsSync(source)) {
  console.error("cesium Build/Cesium was not found. Run npm install in perviy/.");
  process.exit(1);
}

mkdirSync(destination, { recursive: true });

for (const folder of folders) {
  const from = join(source, folder);
  const to = join(destination, folder);
  if (!existsSync(from)) {
    console.error(`Missing Cesium static folder: ${from}`);
    process.exit(1);
  }
  rmSync(to, { recursive: true, force: true });
  cpSync(from, to, { recursive: true });
}

console.log("Copied Cesium Workers, ThirdParty, Assets, and Widgets to public/cesium");
