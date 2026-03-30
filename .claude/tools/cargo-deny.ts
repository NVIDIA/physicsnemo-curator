import { tool } from "@anthropic-ai/tool";
import { z } from "zod";

export default tool({
  description:
    "Run cargo-deny to audit Rust dependencies for license compliance, security advisories, and banned crates.",
  args: z.object({
    check_type: z
      .enum(["licenses", "advisories", "bans", "sources"])
      .optional()
      .describe("Run a specific check only (default: all checks)"),
  }),
  async execute({ check_type }) {
    const args = ["deny", "--manifest-path", "src/rust/Cargo.toml", "check"];
    if (check_type) args.push(check_type);
    const result = Bun.spawnSync(["cargo", ...args], {
      cwd: import.meta.dir + "/../..",
    });
    const stdout = result.stdout.toString();
    const stderr = result.stderr.toString();
    return `Exit code: ${result.exitCode}\n${stdout}\n${stderr}`.trim();
  },
});
