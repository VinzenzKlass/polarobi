polarobi ships an [agent skill](https://agentskills.io/) that teaches coding agents its API, the frame patterns that
replace gurobipy loops, and how to check a ported model. It lives in
[`skills/polarobi`](https://github.com/VinzenzKlass/polarobi/tree/master/skills/polarobi) in the repository and
follows the open Agent Skills format, so any agent that supports skills can use it.

=== "skills CLI"
    The [`skills`](https://github.com/vercel-labs/skills) CLI, which needs Node.js, installs it for the agents it
    finds on your machine and asks which ones to use:
    ```shell
    npx skills add VinzenzKlass/polarobi -g
    ```
    `-g` installs it for your user; leave it out to install it into the current project only. To pick agents
    without the prompt, pass them with `--agent`, for example `--agent claude-code opencode`.
    `npx skills update` picks up later versions of the skill.

=== "Manual"
    Copy the `skills/polarobi` folder into your agent's skills folder:
    ```shell
    git clone --depth 1 https://github.com/VinzenzKlass/polarobi /tmp/polarobi
    mkdir -p ~/.agents/skills
    cp -r /tmp/polarobi/skills/polarobi ~/.agents/skills/
    ```
    Replace `~/.agents/skills` with the folder your agent reads:

    | Agent       | For your user                                    | For one project                         |
    |-------------|--------------------------------------------------|-----------------------------------------|
    | Claude Code | `~/.claude/skills`                               | `.claude/skills`                        |
    | OpenCode    | `~/.config/opencode/skills` or `~/.agents/skills` | `.opencode/skills` or `.agents/skills`  |
    | Others      | see the agent's documentation                    |                                         |

The skill describes the polarobi release it ships with. Update it when you upgrade polarobi.
