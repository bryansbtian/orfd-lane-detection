# Security Policy

## Reporting a Vulnerability

If you believe you have found a security vulnerability in Off-Road Autonomy, please report
it privately. Do not open a public issue, pull request, or discussion, since that could
expose users of the project before a fix is available.

Use GitHub private vulnerability reporting: open the **Security** tab of this repository
and choose **Report a vulnerability**. That channel is private to the maintainers.

Please include:

- A description of the issue and the potential impact.
- Steps to reproduce, or a proof of concept.
- The affected area (module, command, config key, dependency) if known.
- Any relevant logs or output, with secrets and personal data redacted.

## What to Expect

- We aim to acknowledge your report within 3 business days.
- We will investigate, keep you updated on progress, and let you know when a fix ships.
- Please give us a reasonable amount of time to address the issue before any public
  disclosure.

## Supported Versions

This project is under active development. Only the `main` branch receives security fixes.

## Scope

In scope:

- The guardrails that decide what the vehicle does with no one watching: the safe stop,
  the perception gate, speed and steering limits, and parking on shutdown. Anything that
  lets the vehicle keep driving when one of these should have stopped it.
- The BeamNG connection: the client talks to the simulator over an unauthenticated TCP
  socket, so anything that makes it connect to, or accept commands from, a host other
  than the configured one.
- Configuration loading: a YAML file (including one reached through `extends:`) that can
  make the stack execute code or read files outside the repository.
- The container: anything that makes the image run as root, or bakes credentials or model
  files into a layer.
- Dependency issues with a demonstrated, exploitable impact on this project.

Out of scope:

- Reports from automated scanners without a demonstrated, exploitable impact.
- Vulnerabilities in BeamNG.tech, beamngpy, Ultralytics or PyTorch themselves, unless our
  use of them is what creates the vulnerability.
- Exposure caused by running the BeamNG tech socket on an untrusted network; it has no
  authentication, and should only listen on a network you control.
- Denial of service, volumetric, or rate-limit testing.

## Handling Secrets

Never include real secrets, API keys, tokens, or production credentials in a report. If
you discover an exposed secret, tell us what was exposed and where, but do not paste the
value.

The same rule governs this repository: `.env` is git-ignored, and model weights, datasets
and recorded output stay out of git and out of the image. Any file committed by a tool in
this project must be reviewed for credentials and personal data before it lands.
