# Azure DevOps usage

| Azure DevOps feature | How it's used here | Evidence for your resume |
|---|---|---|
| **Azure Boards** | Backlog of epics/stories mapped to the 8-week plan; commits link via `AB#123` | Screenshot of board + burndown |
| **Azure Pipelines** (`azure-pipelines.yml`) | Lint, tests, eval gate, image build, staged deploy | Pipeline run history |
| **Environments + approvals** | `rag-staging` auto, `rag-prod` needs manual approval | Approval log |
| **Variable groups / Key Vault link** | `gcp-rag` group holds project/region/WIF settings (no secrets) | - |
| **Test Plans / test results** | JUnit published each run; eval-results published as artifact | Trend chart |
| **Azure Artifacts** (optional) | Host the built wheel for `ragassist` | Feed screenshot |

## One-time setup
1. Create an Azure DevOps org + project (free for small teams): https://dev.azure.com
2. Pipelines -> New pipeline -> GitHub -> select this repo -> existing YAML `azure-pipelines.yml`.
3. Library -> Variable group `gcp-rag`: `GCP_PROJECT`, `GCP_REGION`, `AR_REPO` (=rag), `WIF_PROVIDER`, `DEPLOY_SA`, `GCP_WIF_CONNECTION_ID`.
4. Pipelines -> Environments: create `rag-staging` and `rag-prod`; add an **Approvals** check to `rag-prod`.
5. Project settings -> Service connections: a Workload Identity Federation connection for GCP; finish the TODO in `infra/terraform/main.tf`.
6. Boards: enable the GitHub integration and use `AB#<id>` in commit messages.

## Caveat
The OIDC-token step in the Build stage is a sketch to verify against current Azure DevOps/GCP WIF docs before first run.
Fallback: a service-account key in a secure variable group (works, but weaker security story).
