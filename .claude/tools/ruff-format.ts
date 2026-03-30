import { tool } from "@anthropic-ai/tool";
import { z } from "zod";

export default tool({
  description:
    "Run the ruff formatter on Python source files. Can format in-place or check formatting without changes.",
  args: z.object({
    paths: z
      .string()
      .default(".")
      .describe("File or directory paths to format (default: current directory)"),
    check: z
      .boolean()
      .default(false)
      .describe("Check formatting without modifying files"),
  }),
  async execute({ paths, check }) {
    const args = ["run", "ruff", "format"];
    if (check) args.push("--check");
    args.push(paths);
    const result = Bun.spawnSync(["uv", ...args], {
      cwd: import.meta.dir + "/../..",
    });
    const stdout = result.stdout.toString();
    const stderr = result.stderr.toString();
    return `Exit code: ${result.exitCode}\n${stdout}\n${stderr}`.trim();
  },
});
