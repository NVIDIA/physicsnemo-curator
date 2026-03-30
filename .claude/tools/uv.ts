import { tool } from "@anthropic-ai/tool";
import { z } from "zod";

export default tool({
  description:
    "Run uv commands for Python environment and dependency management. Supports any uv subcommand (sync, add, remove, lock, run, build, etc.).",
  args: z.object({
    command: z
      .string()
      .describe(
        'The uv subcommand and arguments to run (e.g. "sync --group dev", "add requests", "run pytest")'
      ),
  }),
  async execute({ command }) {
    const args = command.split(/\s+/);
    const result = Bun.spawnSync(["uv", ...args], {
      cwd: import.meta.dir + "/../..",
    });
    const stdout = result.stdout.toString();
    const stderr = result.stderr.toString();
    return `Exit code: ${result.exitCode}\n${stdout}\n${stderr}`.trim();
  },
});
