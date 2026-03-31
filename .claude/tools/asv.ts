import { tool } from "@anthropic-ai/tool";
import { z } from "zod";

export default tool({
  description:
    "Run airspeed velocity (ASV) benchmarks for long-term performance tracking. Supports running benchmarks, publishing results, previewing the dashboard, and comparing revisions.",
  args: z.object({
    command: z
      .enum(["run", "quick", "publish", "preview", "compare", "show", "find"])
      .describe("The ASV subcommand to execute"),
    revisions: z
      .string()
      .optional()
      .describe(
        'Git revision range or refs (e.g. "HEAD^!", "main..feature", "v0.1.0 v0.2.0")'
      ),
    bench: z
      .string()
      .optional()
      .describe(
        'Benchmark filter pattern (e.g. "TimePipeline" to match specific benchmarks)'
      ),
    extra: z
      .string()
      .optional()
      .describe("Additional CLI arguments passed directly to asv"),
  }),
  async execute({ command, revisions, bench, extra }) {
    const args: string[] = ["run", "asv"];
    const cwd = import.meta.dir + "/../..";

    switch (command) {
      case "run":
        args.push("run");
        if (revisions) args.push(revisions);
        else args.push("HEAD^!");
        if (bench) args.push("--bench", bench);
        break;
      case "quick":
        args.push("run", "--quick", "--dry-run");
        if (revisions) args.push(revisions);
        else args.push("HEAD^!");
        if (bench) args.push("--bench", bench);
        break;
      case "publish":
        args.push("publish");
        break;
      case "preview":
        args.push("preview");
        break;
      case "compare":
        args.push("compare");
        if (revisions) {
          const refs = revisions.split(/\s+/);
          args.push(...refs);
        }
        break;
      case "show":
        args.push("show");
        if (revisions) args.push(revisions);
        break;
      case "find":
        args.push("find");
        if (revisions) args.push(revisions);
        if (bench) args.push(bench);
        break;
    }

    if (extra) args.push(...extra.split(/\s+/));

    const result = Bun.spawnSync(["uv", ...args], { cwd });
    const stdout = result.stdout.toString();
    const stderr = result.stderr.toString();
    return `Exit code: ${result.exitCode}\n${stdout}\n${stderr}`.trim();
  },
});
