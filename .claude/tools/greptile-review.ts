import { tool } from "@anthropic-ai/tool";
import { z } from "zod";

export default tool({
  description: `Fetch Greptile AI code review comments from an open PR on NVIDIA/physicsnemo-curator.

Auto-detects the PR number from the current branch (or accepts a PR number directly).
Filters review comments to only those from greptile-apps[bot], extracts priority level,
file path, line number, and the review body, then returns them in a structured format
ready to parse and address one by one.

Requires: gh CLI authenticated with repo scope.`,
  args: z.object({
    pr: z
      .number()
      .optional()
      .describe(
        "PR number. If omitted, auto-detects from the current branch."
      ),
    branch: z
      .string()
      .optional()
      .describe(
        "Branch name to look up. If omitted, uses the current git branch."
      ),
  }),
  async execute({ pr, branch }) {
    const cwd = import.meta.dir + "/../..";
    const repo = "NVIDIA/physicsnemo-curator";

    // --- Resolve PR number ---
    let prNumber = pr;

    if (!prNumber) {
      // Determine branch name
      let branchName = branch;
      if (!branchName) {
        const branchResult = Bun.spawnSync(
          ["git", "rev-parse", "--abbrev-ref", "HEAD"],
          { cwd }
        );
        if (branchResult.exitCode !== 0) {
          return `ERROR: Could not determine current branch:\n${branchResult.stderr.toString()}`;
        }
        branchName = branchResult.stdout.toString().trim();
      }

      // Detect fork owner from remotes
      const remoteResult = Bun.spawnSync(["git", "remote", "-v"], { cwd });
      const remoteOutput = remoteResult.stdout.toString();
      const remoteLines = remoteOutput
        .split("\n")
        .filter((l: string) => l.includes("(push)"));

      let forkOwner: string | null = null;
      for (const line of remoteLines) {
        const match = line.match(
          /^(\S+)\s+(?:git@github\.com:|https:\/\/github\.com\/)([^/]+)\/([^/\s.]+)/
        );
        if (!match) continue;
        const [, , owner] = match;
        if (owner !== "NVIDIA") {
          forkOwner = owner;
          break;
        }
      }

      // If no fork remote found, try gh api for current user
      if (!forkOwner) {
        const whoami = Bun.spawnSync(
          ["gh", "api", "user", "--jq", ".login"],
          { cwd }
        );
        if (whoami.exitCode === 0) {
          forkOwner = whoami.stdout.toString().trim();
        }
      }

      if (!forkOwner) {
        return "ERROR: Could not detect fork owner. Provide --pr number directly.";
      }

      // Search for PR matching this branch
      const head = `${forkOwner}:${branchName}`;
      const searchResult = Bun.spawnSync(
        [
          "gh",
          "api",
          `repos/${repo}/pulls`,
          "--jq",
          `.[] | select(.head.label == "${head}") | .number`,
        ],
        { cwd, timeout: 30_000 }
      );

      if (searchResult.exitCode !== 0) {
        return `ERROR querying PRs:\n${searchResult.stderr.toString()}`;
      }

      const numbers = searchResult.stdout
        .toString()
        .trim()
        .split("\n")
        .filter(Boolean);
      if (numbers.length === 0) {
        return `No open PR found for head '${head}' in ${repo}.\n\nTip: Open a PR first with the open-pr tool, or pass --pr <number> directly.`;
      }
      prNumber = parseInt(numbers[0], 10);
    }

    // --- Fetch all review comments ---
    // GitHub paginates at 30 by default; fetch up to 100
    const commentsResult = Bun.spawnSync(
      [
        "gh",
        "api",
        `repos/${repo}/pulls/${prNumber}/comments`,
        "--paginate",
        "--jq",
        `.[] | select(.user.login == "greptile-apps[bot]") | {path, line, original_line, diff_hunk: .diff_hunk[-200:], body}`,
      ],
      { cwd, timeout: 60_000 }
    );

    if (commentsResult.exitCode !== 0) {
      return `ERROR fetching comments for PR #${prNumber}:\n${commentsResult.stderr.toString()}`;
    }

    const rawComments = commentsResult.stdout.toString().trim();
    if (!rawComments) {
      return `No Greptile comments found on PR #${prNumber}.\n\nThe bot may not have reviewed this PR yet, or all comments have been resolved.`;
    }

    // Parse individual JSON objects (gh --jq outputs one per line)
    const commentObjects = rawComments
      .split("\n")
      .filter(Boolean)
      .map((line: string) => {
        try {
          return JSON.parse(line);
        } catch {
          return null;
        }
      })
      .filter(Boolean);

    // --- Extract priority and clean body ---
    interface GreptileComment {
      path: string;
      line: number | null;
      priority: string;
      title: string;
      body: string;
      suggestion: string | null;
    }

    const parsed: GreptileComment[] = commentObjects.map(
      (c: { path: string; line: number | null; original_line: number | null; body: string }) => {
        let body = c.body;

        // Extract priority from badge image
        let priority = "P?";
        const badgeMatch = body.match(
          /alt="(P[0-3])"/
        );
        if (badgeMatch) {
          priority = badgeMatch[1];
        }

        // Strip HTML badge markup
        body = body.replace(
          /<a[^>]*>.*?<\/a>\s*/gs,
          ""
        );

        // Extract title (bold text at start)
        let title = "";
        const titleMatch = body.match(/^\*\*(.+?)\*\*/);
        if (titleMatch) {
          title = titleMatch[1];
        }

        // Extract suggestion block if present
        let suggestion: string | null = null;
        const suggestionMatch = body.match(
          /```suggestion\n([\s\S]*?)```/
        );
        if (suggestionMatch) {
          suggestion = suggestionMatch[1].trimEnd();
        }

        return {
          path: c.path,
          line: c.line ?? c.original_line,
          priority,
          title,
          body: body.trim(),
          suggestion,
        };
      }
    );

    // Sort by priority (P1 first)
    parsed.sort((a: GreptileComment, b: GreptileComment) => {
      const order: Record<string, number> = { P0: 0, P1: 1, P2: 2, P3: 3, "P?": 4 };
      return (order[a.priority] ?? 4) - (order[b.priority] ?? 4);
    });

    // --- Format output ---
    const lines: string[] = [];
    lines.push(`# Greptile Review: PR #${prNumber}`);
    lines.push(`Found ${parsed.length} comment(s)\n`);

    // Summary table
    const counts: Record<string, number> = {};
    for (const c of parsed) {
      counts[c.priority] = (counts[c.priority] || 0) + 1;
    }
    const countStr = Object.entries(counts)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([p, n]) => `${p}: ${n}`)
      .join(", ");
    lines.push(`Priority breakdown: ${countStr}\n`);

    lines.push("---\n");

    for (let i = 0; i < parsed.length; i++) {
      const c = parsed[i];
      const loc = c.line ? `${c.path}:${c.line}` : c.path;
      lines.push(`## [${c.priority}] ${c.title || "(no title)"}`);
      lines.push(`**File:** \`${loc}\`\n`);
      lines.push(c.body);
      if (c.suggestion) {
        lines.push(`\n**Suggested fix:**\n\`\`\`\n${c.suggestion}\n\`\`\``);
      }
      if (i < parsed.length - 1) {
        lines.push("\n---\n");
      }
    }

    return lines.join("\n");
  },
});
