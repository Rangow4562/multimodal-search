# Security Policy

## Supported Versions

Currently, security updates and patches are provided for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0.x | :x:                |

## Reporting a Vulnerability

We take the security of this project seriously. If you find a security vulnerability, please report it responsibly by following these steps:

1. **Do not open a public issue** on GitHub/GitLab.
2. Email your findings to the maintainer's security email address (e.g., `security@example.com` or your organization's security contact).
3. Provide a detailed description of the vulnerability, including:
   - Steps to reproduce the issue.
   - Potential impact.
   - Any suggested remediations or patches.

We will acknowledge receipt of your report within 48 hours and provide a timeline for addressing the issue.

## Security Best Practices

To keep this RAG pipeline secure, please adhere to the following guidelines:

### 1. API Keys & Secrets Management
- **Never commit secrets**: Do not commit the `.env` file or any hardcoded API keys/connection strings to the repository.
- **Use Environment Variables**: Always load keys (Azure OpenAI, Mistral, Azure AI Search, Azure Storage) using environment variables.
- **Key Rotation**: Rotate your API keys and storage connection strings regularly.

### 2. Dependency Management
- Keep dependencies updated by regularly running `pip install --upgrade -r requirements.txt`.
- Scan dependencies for known vulnerabilities using tools like `safety` or `pip-audit`.

### 3. Data Privacy
- **Private Blob Storage**: Ensure your Azure Storage containers containing source PDFs and extracted images are configured with private access levels.
- **Access Control**: Use Azure Managed Identities or Role-Based Access Control (RBAC) instead of raw admin keys and connection strings in production environments.
