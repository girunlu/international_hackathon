import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import fs from "fs";
import { promises as fsPromises } from "fs";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
const staticFilesDir = path.resolve(__dirname, "static_files");

async function copyDirectory(source: string, destination: string) {
  await fsPromises.mkdir(destination, { recursive: true });
  const entries = await fsPromises.readdir(source, { withFileTypes: true });

  for (const entry of entries) {
    const sourcePath = path.join(source, entry.name);
    const destinationPath = path.join(destination, entry.name);

    if (entry.isDirectory()) {
      await copyDirectory(sourcePath, destinationPath);
    } else if (entry.isFile()) {
      await fsPromises.copyFile(sourcePath, destinationPath);
    }
  }
}

function staticFilesPlugin() {
  let resolvedOutDir = "dist";

  return {
    name: "serve-static-files",
    configResolved(config) {
      resolvedOutDir = config.build.outDir;
    },
    configureServer(server) {
      server.middlewares.use("/static_files", (req, res, next) => {
        if (!req.url) {
          next();
          return;
        }

        const requestedPath = decodeURIComponent(req.url.split("?")[0] ?? "");
        const normalized = requestedPath.replace(/^\/+/, "");
        const filePath = path.join(staticFilesDir, normalized);

        if (!filePath.startsWith(staticFilesDir)) {
          next();
          return;
        }

        fs.readFile(filePath, (err, data) => {
          if (err) {
            next();
            return;
          }

          res.setHeader("Content-Type", "application/json");
          res.setHeader("Cache-Control", "no-store, max-age=0");
          res.end(data);
        });
      });
    },
    async closeBundle() {
      const destination = path.join(resolvedOutDir, "static_files");
      await fsPromises.rm(destination, { recursive: true, force: true });
      await copyDirectory(staticFilesDir, destination);
    },
  };
}

export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
  },
  plugins: [react(), staticFilesPlugin(), mode === "development" && componentTagger()].filter(
    Boolean,
  ),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
