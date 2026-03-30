import { tool } from "@anthropic-ai/tool";
import { z } from "zod";

export default tool({
  description:
    "Run cargo clippy to lint Rust source code. Returns warnings and errors. Optionally auto-fixes issues.",
  args: z.object({
    fix: z
      .boolean()
      .default(false)
      .describe("Automatically apply suggested fixes"),
  }),
  async execute({ fix }) {
    const args = ["clippy", "--manifest-path", "src/rust/Cargo.toml"];
    if (fix) args.push("--fix", "--allow-dirty");
    args.push("--", "-D", "warnings");
    const result = Bun.spawnSync(["cargo", ...args], {
      cwd: import.meta.dir + "/../..",
    });
    const stdout = result.stdout.toString();
    const stderr = result.stderr.toString();
    return `Exit code: ${result.exitCode}\n${stdout}\n${stderr}`.trim();
  },
});
