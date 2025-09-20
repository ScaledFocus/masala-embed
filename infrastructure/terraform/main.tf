terraform {
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 5.0"
    }
  }
}

variable "cloudflare_api_key" {
  type      = string
  sensitive = true
}

provider "cloudflare" {
  api_token = var.cloudflare_api_key
}

variable "bucket_name" {
  type    = string
  default = "masala-embed-models"
}

variable "cloudflare_account_id" {
  type = string
}

resource "cloudflare_r2_bucket" "models" {
  account_id = var.cloudflare_account_id
  name       = var.bucket_name
}
