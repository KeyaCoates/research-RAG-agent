terraform {
  required_providers { google = { source = "hashicorp/google", version = "~> 6.0" } }
}
variable "project" { type = string }
variable "region"  { type = string, default = "us-central1" }

provider "google" {
  project = var.project
  region  = var.region
}

resource "google_project_service" "apis" {
  for_each = toset(["run.googleapis.com", "aiplatform.googleapis.com", "artifactregistry.googleapis.com",
                    "cloudbuild.googleapis.com", "iamcredentials.googleapis.com", "cloudtrace.googleapis.com"])
  service = each.key
}

resource "google_artifact_registry_repository" "rag" {
  repository_id = "rag"
  format        = "DOCKER"
  location      = var.region
  depends_on    = [google_project_service.apis]
}

resource "google_storage_bucket" "docs" {
  name                        = "${var.project}-rag-docs"
  location                    = var.region
  uniform_bucket_level_access = true
}

resource "google_service_account" "runtime" { account_id = "rag-runtime" }
resource "google_service_account" "deployer" { account_id = "rag-deployer" }

resource "google_project_iam_member" "runtime_roles" {
  for_each = toset(["roles/aiplatform.user", "roles/cloudtrace.agent", "roles/storage.objectViewer"])
  project  = var.project
  role     = each.key
  member   = "serviceAccount:${google_service_account.runtime.email}"
}

# TODO: workload identity pool + provider trusting Azure DevOps' OIDC issuer
# (https://vstoken.dev.azure.com/<org-id>), and roles for the deployer SA
# (run.admin, artifactregistry.writer, cloudbuild.builds.editor, iam.serviceAccountUser).
