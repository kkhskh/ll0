import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Cesium's workers and assets are copied to public/cesium. Turbopack does
  // not run CopyWebpackPlugin, so that copy is a predev/prebuild script.
  agentRules: false,
  serverExternalPackages: ["cesium"],
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
