import { tool } from "@anthropic-ai/tool";
import { z } from "zod";

export default tool({
  description:
    "Run cargo fmt to format Rust source code. Can format in-place or check formatting.",
  args: z.object({
    check: z
      .boolean()
      .default(false)
      .describe("Check formatting without modifying files"),
  }),
  async execute({ check }) {
    const args = ["fmt", "--manifest-path", "src/rust/Cargo.toml"];
    if (check) args.push("--", "--check");
    const result = Bun.spawnSync(["cargo", ...args], {
      cwd: import.meta.dir + "/../..",
    });
    const stdout = result.stdout.toString();
    const stderr = result.stderr.toString();
    return `Exit code: ${result.exitCode}\n${stdout}\n${stderr}`.trim();
  },
});
