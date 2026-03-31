import { tool } from "@anthropic-ai/tool";
import { z } from "zod";

export default tool({
  description:
    "Build Sphinx documentation. Returns build output including any warnings or errors.",
  args: z.object({
    warningsAsErrors: z
      .boolean()
      .default(true)
      .describe("Treat warnings as errors (-W flag)"),
    clean: z
      .boolean()
      .default(false)
      .describe("Remove the build directory before building"),
  }),
  async execute({ warningsAsErrors, clean }) {
    const root = import.meta.dir + "/../..";
    if (clean) {
      Bun.spawnSync(["rm", "-rf", "docs/_build"], { cwd: root });
    }
    const args = [
      "run",
      "--group",
      "docs",
      "sphinx-build",
    ];
    if (warningsAsErrors) args.push("-W");
    args.push("docs", "docs/_build/html");
    const result = Bun.spawnSync(["uv", ...args], { cwd: root });
    const stdout = result.stdout.toString();
    const stderr = result.stderr.toString();
    return `Exit code: ${result.exitCode}\n${stdout}\n${stderr}`.trim();
  },
});
