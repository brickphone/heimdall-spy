import type { NextConfig } from "next";

// The FastAPI service that does all inference. Read at build time (rewrites are
// serialised into the build), so the Docker image takes it as a build argument.
const apiUrl = process.env.SWESPY_API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  // Long-running Node server (`node server.js`); no serverless functions.
  output: "standalone",
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
  // The browser calls /api/* on this origin; the Node server forwards it unchanged to
  // the public FastAPI endpoints. No inference or business logic lives in this app.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${apiUrl}/:path*` }];
  },
  experimental: {
    // CPU-only containers take ~20 s per photo; the default proxy timeout is 30 s.
    proxyTimeout: 120_000,
  },
  poweredByHeader: false,
};

export default nextConfig;
