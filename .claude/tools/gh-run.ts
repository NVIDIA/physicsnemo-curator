import { tool } from "@anthropic-ai/tool";
import { z } from "zod";

export default tool({
  description: `Fetch a GitHub Actions workflow run and return a structured summary of all jobs.

Takes a run ID (the numeric database ID from the Actions tab URL) and retrieves
every job in that run. For each job, reports status/conclusion, duration, and
step-by-step results. For **failed** jobs, automatically fetches the full job
log and extracts the failing step's output so the agent can diagnose the issue
without leaving the conversation.

Can also list recent runs when called without an ID, and supports filtering
by workflow name and branch.

Requires: gh CLI authenticated with repo scope.`,
  args: z.object({
    run_id: z
      .number()
      .optional()
      .describe(
        "Workflow run database ID (from the URL or gh run list). " +
          "If omitted, lists recent runs instead."
      ),
    workflow: z
      .string()
      .optional()
      .describe(
        "Filter by workflow name when listing runs (e.g. 'Lint', 'Build & Install'). " +
          "Ignored when run_id is provided."
      ),
    branch: z
      .string()
      .optional()
      .describe(
        "Filter by branch when listing runs (e.g. 'main', 'refactor'). " +
          "Ignored when run_id is provided."
      ),
    limit: z
      .number()
      .default(10)
      .describe("Number of recent runs to list (default 10). Ignored when run_id is provided."),
    log_lines: z
      .number()
      .default(80)
      .describe(
        "Max lines of log output to include per failed step (default 80). " +
          "Set higher for verbose build failures."
      ),
  }),
  async execute({ run_id, workflow, branch, limit, log_lines }) {
    const repo = "NVIDIA/physicsnemo-curator";

    // ---------------------------------------------------------------
    // MODE 1: List recent runs
    // ---------------------------------------------------------------
    if (!run_id) {
      const args = [
        "gh",
        "run",
        "list",
        "--repo",
        repo,
        "--limit",
        String(limit),
        "--json",
        "databaseId,displayTitle,status,conclusion,workflowName,headBranch,event,createdAt,number",
      ];
      if (workflow) {
        args.push("--workflow", workflow);
      }
      if (branch) {
        args.push("--branch", branch);
      }

      const result = Bun.spawnSync(args, { timeout: 30_000 });
      if (result.exitCode !== 0) {
        return `ERROR listing runs:\n${result.stderr.toString()}`;
      }

      const runs = JSON.parse(result.stdout.toString());
      if (runs.length === 0) {
        return "No workflow runs found matching the given filters.";
      }

      const lines: string[] = [];
      lines.push(`# Recent Workflow Runs (${repo})\n`);

      // Summary counts
      const statusCounts: Record<string, number> = {};
      for (const r of runs) {
        const key = r.conclusion || r.status;
        statusCounts[key] = (statusCounts[key] || 0) + 1;
      }
      lines.push(
        `Total: ${runs.length} | ` +
          Object.entries(statusCounts)
            .map(([k, v]) => `${k}: ${v}`)
            .join(", ") +
          "\n"
      );

      // Table
      lines.push("| Run ID | Workflow | Branch | Status | Title |");
      lines.push("|--------|----------|--------|--------|-------|");
      for (const r of runs) {
        const status =
          r.status === "completed"
            ? r.conclusion === "success"
              ? "PASS"
              : r.conclusion === "failure"
                ? "FAIL"
                : r.conclusion?.toUpperCase() || "?"
            : r.status.toUpperCase();
        lines.push(
          `| ${r.databaseId} | ${r.workflowName} | ${r.headBranch} | ${status} | ${r.displayTitle.substring(0, 60)} |`
        );
      }

      lines.push(
        "\nUse this tool again with a specific `run_id` to see job details and failure logs."
      );
      return lines.join("\n");
    }

    // ---------------------------------------------------------------
    // MODE 2: Fetch run details + jobs
    // ---------------------------------------------------------------

    // Fetch run metadata
    const runResult = Bun.spawnSync(
      [
        "gh",
        "api",
        `repos/${repo}/actions/runs/${run_id}`,
        "--jq",
        "{id,name,status,conclusion,html_url,head_branch,event,created_at,updated_at,display_title}",
      ],
      { timeout: 30_000 }
    );

    if (runResult.exitCode !== 0) {
      return `ERROR fetching run ${run_id}:\n${runResult.stderr.toString()}`;
    }

    const run = JSON.parse(runResult.stdout.toString());

    // Fetch all jobs
    const jobsResult = Bun.spawnSync(
      ["gh", "run", "view", String(run_id), "--repo", repo, "--json", "jobs"],
      { timeout: 60_000 }
    );

    if (jobsResult.exitCode !== 0) {
      return `ERROR fetching jobs for run ${run_id}:\n${jobsResult.stderr.toString()}`;
    }

    const { jobs } = JSON.parse(jobsResult.stdout.toString());

    // Categorize jobs
    interface Job {
      databaseId: number;
      name: string;
      status: string;
      conclusion: string;
      startedAt: string;
      completedAt: string;
      url: string;
      steps: Array<{
        name: string;
        number: number;
        status: string;
        conclusion: string;
      }>;
    }

    const passed: Job[] = [];
    const failed: Job[] = [];
    const cancelled: Job[] = [];
    const skipped: Job[] = [];
    const running: Job[] = [];

    for (const job of jobs as Job[]) {
      if (job.status !== "completed") {
        running.push(job);
      } else if (job.conclusion === "success") {
        passed.push(job);
      } else if (job.conclusion === "failure") {
        failed.push(job);
      } else if (job.conclusion === "cancelled") {
        cancelled.push(job);
      } else {
        skipped.push(job);
      }
    }

    // Build output
    const lines: string[] = [];

    // Header
    const overallStatus =
      run.status === "completed"
        ? run.conclusion === "success"
          ? "PASSED"
          : "FAILED"
        : "IN PROGRESS";

    lines.push(`# Workflow Run: ${run.name} [${overallStatus}]`);
    lines.push(`**Run ID:** ${run.id}`);
    lines.push(`**Branch:** ${run.head_branch} (${run.event})`);
    lines.push(`**Title:** ${run.display_title}`);
    lines.push(`**URL:** ${run.html_url}`);
    lines.push(`**Created:** ${run.created_at}\n`);

    // Job summary
    lines.push("## Job Summary\n");
    lines.push(
      `Total: ${jobs.length} | ` +
        `Passed: ${passed.length} | ` +
        `Failed: ${failed.length} | ` +
        `Cancelled: ${cancelled.length} | ` +
        `Skipped: ${skipped.length} | ` +
        `Running: ${running.length}\n`
    );

    // Passed jobs (brief)
    if (passed.length > 0) {
      lines.push("### Passed Jobs\n");
      for (const job of passed) {
        const duration = _duration(job.startedAt, job.completedAt);
        lines.push(`- **${job.name}** (${duration})`);
      }
      lines.push("");
    }

    // Running jobs
    if (running.length > 0) {
      lines.push("### Running Jobs\n");
      for (const job of running) {
        lines.push(`- **${job.name}** — ${job.status}`);
      }
      lines.push("");
    }

    // Cancelled jobs
    if (cancelled.length > 0) {
      lines.push("### Cancelled Jobs\n");
      for (const job of cancelled) {
        lines.push(`- **${job.name}**`);
      }
      lines.push("");
    }

    // Failed jobs (detailed with logs)
    if (failed.length > 0) {
      lines.push("---\n");
      lines.push("## Failed Jobs (detailed)\n");

      for (const job of failed) {
        const duration = _duration(job.startedAt, job.completedAt);
        lines.push(`### ${job.name} (${duration})\n`);
        lines.push(`**Job URL:** ${job.url}\n`);

        // Steps table
        lines.push("| # | Step | Result |");
        lines.push("|---|------|--------|");

        const failedSteps: string[] = [];
        for (const step of job.steps) {
          // Skip internal GH steps (Set up job, Post *, Complete job)
          if (
            step.name === "Set up job" ||
            step.name === "Complete job" ||
            step.name.startsWith("Post ")
          ) {
            continue;
          }
          const icon =
            step.conclusion === "success"
              ? "PASS"
              : step.conclusion === "failure"
                ? "**FAIL**"
                : step.conclusion === "skipped"
                  ? "SKIP"
                  : step.conclusion?.toUpperCase() || step.status;
          lines.push(`| ${step.number} | ${step.name} | ${icon} |`);

          if (step.conclusion === "failure") {
            failedSteps.push(step.name);
          }
        }
        lines.push("");

        // Fetch logs for this failed job
        const logResult = Bun.spawnSync(
          [
            "gh",
            "api",
            `repos/${repo}/actions/jobs/${job.databaseId}/logs`,
          ],
          { timeout: 60_000 }
        );

        if (logResult.exitCode === 0) {
          const fullLog = logResult.stdout.toString();

          // Extract logs around the error
          const logLines = fullLog.split("\n");

          // Find lines with ##[error] markers
          const errorLines: number[] = [];
          for (let i = 0; i < logLines.length; i++) {
            if (logLines[i].includes("##[error]")) {
              errorLines.push(i);
            }
          }

          if (errorLines.length > 0) {
            // Show context around errors
            lines.push("**Error Output:**\n");
            lines.push("```");

            const shown = new Set<number>();
            const contextBefore = Math.floor(log_lines * 0.6);
            const contextAfter = Math.floor(log_lines * 0.2);

            for (const errIdx of errorLines) {
              const start = Math.max(0, errIdx - contextBefore);
              const end = Math.min(logLines.length - 1, errIdx + contextAfter);

              for (let i = start; i <= end; i++) {
                if (!shown.has(i)) {
                  shown.add(i);
                  // Strip timestamp prefix for readability
                  const cleaned = logLines[i].replace(
                    /^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s*/,
                    ""
                  );
                  lines.push(cleaned);
                }
              }

              // Limit total output
              if (shown.size > log_lines) break;
            }

            lines.push("```\n");
          } else {
            // No explicit error markers — show the last N lines
            lines.push("**Log tail (no ##[error] markers found):**\n");
            lines.push("```");
            const tail = logLines.slice(-log_lines);
            for (const line of tail) {
              const cleaned = line.replace(
                /^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s*/,
                ""
              );
              lines.push(cleaned);
            }
            lines.push("```\n");
          }
        } else {
          lines.push(
            `*Could not fetch logs: ${logResult.stderr.toString().trim()}*\n`
          );
        }
      }
    }

    // Final verdict
    if (failed.length === 0 && running.length === 0) {
      lines.push("---\n**All jobs passed.**");
    } else if (failed.length > 0) {
      lines.push("---\n");
      lines.push("## Action Items\n");
      for (const job of failed) {
        const failedStepNames = job.steps
          .filter((s) => s.conclusion === "failure")
          .map((s) => s.name);
        lines.push(
          `- **${job.name}**: Fix failing step(s): ${failedStepNames.join(", ")}`
        );
      }
    }

    return lines.join("\n");
  },
});

function _duration(start: string, end: string): string {
  if (!start || !end) return "?";
  const ms = new Date(end).getTime() - new Date(start).getTime();
  const secs = Math.floor(ms / 1000);
  if (secs < 60) return `${secs}s`;
  const mins = Math.floor(secs / 60);
  const remSecs = secs % 60;
  return `${mins}m${remSecs}s`;
}
