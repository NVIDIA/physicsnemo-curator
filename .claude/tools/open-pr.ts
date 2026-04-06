import { tool } from "@anthropic-ai/tool";
import { z } from "zod";

export default tool({
  description: `Open a pull request from the current branch to NVIDIA/physicsnemo-curator.

Automatically:
  - Auto-detects the fork owner from git remotes
  - Pushes the current branch to the fork remote
  - Creates a PR using the repo's PR template
  - Fills in the description, checklist, and dependencies sections

The PR is always opened against the 'main' branch of NVIDIA/physicsnemo-curator
unless a different base is specified.

Requires: gh CLI authenticated with a token that has repo scope.`,
  args: z.object({
    title: z
      .string()
      .describe(
        "PR title in Conventional Commits style, e.g. 'feat(mesh): add NavierStokesCylinderSource'"
      ),
    description: z
      .string()
      .describe(
        "Standalone description of the changes. Reference issues with 'closes #N'."
      ),
    dependencies: z
      .string()
      .default("None")
      .describe("New dependencies added by this PR, or 'None'"),
    checklist: z
      .object({
        contributing: z
          .boolean()
          .default(true)
          .describe("Familiar with Contributing Guidelines"),
        tests: z
          .boolean()
          .default(true)
          .describe("New or existing tests cover changes"),
        docs: z
          .boolean()
          .default(false)
          .describe("Documentation is up to date"),
        issue: z
          .boolean()
          .default(false)
          .describe("An issue is linked to this PR"),
      })
      .default({})
      .describe("Checklist items to mark as complete"),
    base: z
      .string()
      .default("main")
      .describe("Base branch to merge into (default: main)"),
    draft: z
      .boolean()
      .default(false)
      .describe("Open as a draft PR"),
  }),
  async execute({ title, description, dependencies, checklist, base, draft }) {
    const cwd = import.meta.dir + "/../..";

    // --- Detect fork owner from git remotes ---
    const remoteResult = Bun.spawnSync(["git", "remote", "-v"], { cwd });
    if (remoteResult.exitCode !== 0) {
      return `ERROR: Could not read git remotes:\n${remoteResult.stderr.toString()}`;
    }
    const remoteOutput = remoteResult.stdout.toString();

    // Get current branch name
    const branchResult = Bun.spawnSync(
      ["git", "rev-parse", "--abbrev-ref", "HEAD"],
      { cwd }
    );
    if (branchResult.exitCode !== 0) {
      return `ERROR: Could not determine current branch:\n${branchResult.stderr.toString()}`;
    }
    const branch = branchResult.stdout.toString().trim();

    if (branch === "main" || branch === "master") {
      return `ERROR: You are on '${branch}'. Create a feature branch first.`;
    }

    // Parse remotes to find the fork (non-NVIDIA remote)
    // and the upstream (NVIDIA remote)
    const remoteLines = remoteOutput.split("\n").filter((l: string) => l.includes("(push)"));
    let forkRemote: string | null = null;
    let upstreamRemote: string | null = null;
    let forkOwner: string | null = null;

    for (const line of remoteLines) {
      const match = line.match(
        /^(\S+)\s+(?:git@github\.com:|https:\/\/github\.com\/)([^/]+)\/([^/\s.]+)/
      );
      if (!match) continue;
      const [, remoteName, owner] = match;
      if (owner === "NVIDIA") {
        upstreamRemote = remoteName;
      } else {
        forkRemote = remoteName;
        forkOwner = owner;
      }
    }

    // If only one remote and it's NVIDIA, the user's fork IS origin
    // (they cloned their fork as origin). Detect via gh api.
    if (!forkRemote && upstreamRemote) {
      // All remotes point to NVIDIA — try to use origin and detect user
      const whoami = Bun.spawnSync(
        ["gh", "api", "user", "--jq", ".login"],
        { cwd }
      );
      if (whoami.exitCode === 0) {
        forkOwner = whoami.stdout.toString().trim();
        forkRemote = upstreamRemote;
      } else {
        return `ERROR: All remotes point to NVIDIA/physicsnemo-curator and could not detect your GitHub username. Add your fork as a remote:\n  git remote add fork git@github.com:YOUR_USER/physicsnemo-curator.git`;
      }
    }

    if (!forkRemote || !forkOwner) {
      return `ERROR: Could not detect fork remote. Found remotes:\n${remoteOutput}\n\nAdd your fork:\n  git remote add fork git@github.com:YOUR_USER/physicsnemo-curator.git`;
    }

    // --- Push branch to fork ---
    const pushResult = Bun.spawnSync(
      ["git", "push", "-u", forkRemote, branch],
      { cwd, timeout: 120_000 }
    );
    if (pushResult.exitCode !== 0) {
      const pushErr = pushResult.stderr.toString();
      // Allow "already up to date" style messages
      if (!pushErr.includes("Everything up-to-date") && !pushErr.includes("already exists")) {
        return `ERROR pushing to ${forkRemote}/${branch}:\n${pushErr}`;
      }
    }

    // --- Build PR body from template ---
    const check = (done: boolean) => (done ? "[x]" : "[ ]");
    const body = `<!-- markdownlint-disable MD013-->
# PhysicsNeMo Curator Pull Request

## Description

${description}

## Checklist

- ${check(checklist.contributing)} I am familiar with the [Contributing Guidelines](https://github.com/NVIDIA/physicsnemo-curator/blob/main/CONTRIBUTING.md).
- ${check(checklist.tests)} New or existing tests cover these changes.
- ${check(checklist.docs)} The documentation is up to date with these changes.
- ${check(checklist.issue)} An [issue](https://github.com/NVIDIA/physicsnemo-curator/issues) is linked to this pull request.

## Dependencies

${dependencies}
`;

    // --- Create PR via gh ---
    const prArgs = [
      "gh",
      "pr",
      "create",
      "--repo",
      "NVIDIA/physicsnemo-curator",
      "--title",
      title,
      "--body",
      body,
      "--base",
      base,
      "--head",
      `${forkOwner}:${branch}`,
    ];
    if (draft) prArgs.push("--draft");

    const prResult = Bun.spawnSync(prArgs, { cwd, timeout: 60_000 });
    const prOut = prResult.stdout.toString().trim();
    const prErr = prResult.stderr.toString().trim();

    if (prResult.exitCode !== 0) {
      return `ERROR creating PR:\n${prOut}\n${prErr}`;
    }

    return `Pull request created successfully!\n\n${prOut}\n\nBranch: ${forkOwner}:${branch} → NVIDIA/physicsnemo-curator:${base}${draft ? " (draft)" : ""}`;
  },
});
