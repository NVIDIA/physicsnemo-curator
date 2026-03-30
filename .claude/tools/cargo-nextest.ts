import { tool } from "@anthropic-ai/tool";
import { z } from "zod";

export default tool({
  description:
    "Run Rust tests using cargo-nextest. Each test runs in its own process for better isolation and parallelism.",
  args: z.object({
    filter: z
      .string()
      .optional()
      .describe("Test name filter expression (e.g. 'test_parse' to run matching tests)"),
    retries: z
      .number()
      .default(0)
      .describe("Number of retries for flaky tests"),
  }),
  async execute({ filter, retries }) {
    const args = ["nextest", "run", "--manifest-path", "src/rust/Cargo.toml"];
    if (filter) args.push("-E", filter);
    if (retries > 0) args.push("--retries", String(retries));
    const result = Bun.spawnSync(["cargo", ...args], {
      cwd: import.meta.dir + "/../..",
    });
    const stdout = result.stdout.toString();
    const stderr = result.stderr.toString();
    return `Exit code: ${result.exitCode}\n${stdout}\n${stderr}`.trim();
  },
});
