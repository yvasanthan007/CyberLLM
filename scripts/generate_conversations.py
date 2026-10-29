"""
Generate a synthetic cybersecurity conversational dataset (JSONL).

Categories covered:
  greetings, ethical_hacking, networking, phishing_malware,
  passwords_auth, encryption, web_security, linux_tools,
  incident_response, privacy_risk

Reproducible: seeded RNG. Configurable count via --num-per-category.
Output schema: {"instruction": str, "input": "", "output": str, "category": str}
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "conversation" / "raw_synthetic_conversations.jsonl"

CATEGORIES = {}

# ------------------------- greetings -------------------------
CATEGORIES["greetings"] = [
    ("Hello! What can you help me with?",
     "Hello! I'm CyberLLM, a cybersecurity assistant. I can explain security "
     "concepts, help you understand threats like phishing or malware, describe "
     "defensive practices, and answer questions about ethical hacking, "
     "networking, cryptography, and incident response. What would you like to know?"),
    ("Who are you?",
     "I'm CyberLLM, a conversational cybersecurity assistant. I was built to "
     "answer cybersecurity questions clearly and accurately for learners, "
     "analysts, and defenders. I focus on defensive and educational topics."),
    ("What can you do?",
     "I can explain cybersecurity concepts, walk through how attacks like "
     "phishing or ransomware work at a conceptual level, describe defensive "
     "controls, clarify terminology such as zero-day or CVE, and help with "
     "secure practices for passwords, encryption, and incident response."),
    ("Are you a security scanner?",
     "No. I'm a conversational assistant, not a scanner. I explain security "
     "concepts and best practices. For actual scanning or URL reputation "
     "checks, use dedicated tools — this project also ships a separate "
     "phishing URL classifier endpoint for that purpose."),
    ("Hi, can you teach me cybersecurity?",
     "Absolutely — that's what I'm here for. We can start with fundamentals "
     "like the CIA triad, then move into topics like networking, "
     "authentication, malware, web security, and incident response. Just ask "
     "about any topic that interests you."),
    ("What is cybersecurity?",
     "Cybersecurity is the practice of protecting systems, networks, "
     "programs, and data from digital attacks, damage, or unauthorized "
     "access. It combines technology, processes, and people-focused controls "
     "to preserve confidentiality, integrity, and availability."),
]

# ------------------------- ethical hacking -------------------------
CATEGORIES["ethical_hacking"] = [
    ("What is ethical hacking?",
     "Ethical hacking is the authorized process of identifying and assessing "
     "security vulnerabilities in systems and networks to help protect them. "
     "Ethical hackers operate with explicit permission, follow defined rules "
     "of engagement, and report findings so organizations can fix weaknesses "
     "before malicious attackers exploit them."),
    ("How is ethical hacking different from malicious hacking?",
     "The key difference is authorization and intent. Ethical hackers have "
     "written permission from the system owner, work within agreed scope and "
     "rules of engagement, and document and report what they find. Malicious "
     "hackers exploit systems without permission for personal gain, damage, "
     "or disruption, which is illegal in most jurisdictions."),
    ("What is penetration testing?",
     "Penetration testing is a simulated, authorized attack against a system "
     "or network to evaluate its security. Testers attempt to find and "
     "exploit vulnerabilities the way a real attacker would, then produce a "
     "report with findings, impact, and remediation recommendations. Common "
     "phases include reconnaissance, scanning, exploitation, and reporting."),
    ("What are the phases of a penetration test?",
     "A typical penetration test has these phases: 1) scoping and "
     "authorization, 2) reconnaissance (passive and active information "
     "gathering), 3) scanning and enumeration, 4) vulnerability analysis, "
     "5) exploitation, 6) post-exploitation (assessing impact and "
     "persistence), and 7) reporting and remediation guidance."),
    ("What certifications exist for ethical hackers?",
     "Well-known certifications include CEH (Certified Ethical Hacker), "
     "OSEP and OSCP from Offensive Security, GPEN from GIAC/SANS, and "
     "eJPT from INE. Each has a different focus: CEH is broad and "
     "knowledge-based, OSCP is hands-on and exam-driven, and GPEN focuses on "
     "professional penetration testing methodology."),
    ("Is ethical hacking legal?",
     "Yes, when performed with the explicit, documented authorization of the "
     "system owner and within the agreed scope. Without authorization, the "
     "same activities can violate computer fraud and abuse laws. Ethical "
     "hackers typically work under contracts, rules of engagement, and "
     "sometimes safe-harbor clauses that define what is permitted."),
    ("What is bug bounty hunting?",
     "Bug bounty hunting is security research in which individuals test "
     "organizations' publicly defined scopes — often web applications or "
     "APIs — and are rewarded for responsibly disclosing valid "
     "vulnerabilities. Platforms like HackerOne and Bugcrowd host programs. "
     "Researchers must stay within each program's published scope and rules."),
    ("What is recon in hacking?",
     "Reconnaissance (recon) is the information-gathering phase of a security "
     "assessment. Passive recon collects publicly available data without "
     "touching the target directly (DNS records, search engines, certificate "
     "transparency logs). Active recon interacts with the target, for example "
     "port scanning or enumerating web endpoints."),
]

# ------------------------- networking -------------------------
CATEGORIES["networking"] = [
    ("What is a firewall?",
     "A firewall is a security system that monitors and controls network "
     "traffic based on predefined rules. It sits between trusted and "
     "untrusted networks and permits or blocks packets by address, port, "
     "protocol, or state. Modern firewalls add application-layer inspection "
     "and intrusion prevention."),
    ("What is the difference between TCP and UDP?",
     "TCP is connection-oriented and reliable: it establishes a handshake, "
     "guarantees ordering, and retransmits lost segments, which suits web, "
     "email, and file transfer. UDP is connectionless and lightweight with no "
     "delivery guarantee, which suits real-time traffic like DNS lookups, "
     "voice, and video streaming."),
    ("What is a VPN?",
     "A VPN (virtual private network) creates an encrypted tunnel between a "
     "device and a VPN server so traffic is protected from local network "
     "eavesdropping and the destination sees the VPN's IP rather than the "
     "user's. VPNs improve privacy on untrusted networks but do not by "
     "themselves make a device secure or anonymous."),
    ("What is DNS and why does it matter for security?",
     "DNS translates domain names into IP addresses. It matters for security "
     "because attackers abuse it: DNS spoofing/poisoning can redirect users "
     "to malicious sites, DNS tunneling can exfiltrate data, and "
     "subdomain takeover can hijack records. Defensive measures include "
     "DNSSEC, monitoring, filtering malicious domains, and restricting "
     "recursive resolvers."),
    ("What is a port scan?",
     "A port scan probes a host's network ports to discover which services "
     "are listening. Attackers use it for reconnaissance to find attack "
     "surface; defenders use it to verify that only intended services are "
     "exposed. Tools like Nmap are standard. Unexpected scanning of your "
     "network is a common early reconnaissance indicator."),
    ("What is the principle of least privilege?",
     "Least privilege means every user, process, or service gets only the "
     "minimum access needed to perform its function — nothing more. It "
     "limits the blast radius of compromised accounts, reduces lateral "
     "movement, and is a core principle in zero-trust architecture."),
    ("What is zero trust?",
     "Zero trust is a security model that assumes no implicit trust based on "
     "network location. Every request is authenticated, authorized, and "
     "encrypted regardless of where it originates. It emphasizes strong "
     "identity, device posture checks, least privilege, and continuous "
     "verification — often summarized as 'never trust, always verify'."),
    ("What is network segmentation?",
     "Network segmentation divides a network into isolated zones so that a "
     "compromise in one segment cannot freely reach others. For example, "
     "keeping guest Wi-Fi, servers, and corporate workstations separated "
     "with firewall rules between them limits lateral movement during an "
     "intrusion."),
]

# ------------------------- phishing / malware -------------------------
CATEGORIES["phishing_malware"] = [
    ("What is phishing?",
     "Phishing is a social engineering attack in which an attacker impersonates "
     "a trusted party — by email, SMS, voice, or chat — to trick a victim into "
     "revealing credentials, clicking a malicious link, or opening a malicious "
     "attachment. Defenses include user awareness, email authentication "
     "(SPF, DKIM, DMARC), URL filtering, and multi-factor authentication."),
    ("How can I spot a phishing email?",
     "Look for urgency and pressure tactics, mismatched or lookalike sender "
     "domains, unexpected attachments or login links, generic greetings, and "
     "requests for credentials or payment changes. Hover over links to check "
     "the real destination before clicking. When in doubt, verify through a "
     "separate, known-good channel rather than replying."),
    ("What is the difference between malware, a virus, and a worm?",
     "Malware is the umbrella term for malicious software. A virus infects "
     "other files and needs user action to spread. A worm self-replicates "
     "across networks without user action. Other malware types include "
     "trojans (disguised as legitimate software), ransomware, spyware, "
     "and rootkits."),
    ("What is ransomware?",
     "Ransomware is malware that encrypts a victim's files or locks systems "
     "and demands payment, usually in cryptocurrency, for restoration. "
     "Modern ransomware groups also exfiltrate data and threaten to leak it. "
     "Defense relies on offline and tested backups, patching, email "
     "filtering, MFA, network segmentation, and a rehearsed response plan."),
    ("What is social engineering?",
     "Social engineering is manipulating people — rather than technology — to "
     "gain access or information. Common techniques include phishing, "
     "pretexting (invented scenarios), baiting (infected USB drives), and "
     "tailgating into physical spaces. Awareness training, verification "
     "procedures, and a reporting culture are the primary defenses."),
    ("What is a zero-day vulnerability?",
     "A zero-day is a vulnerability that is publicly known or actively "
     "exploited before a vendor patch exists — the vendor has had 'zero days' "
     "to fix it. Defenses include defense in depth, behavioral detection, "
     "network segmentation, rapid patching once fixes ship, and threat "
     "intelligence feeds."),
    ("What is spear phishing?",
     "Spear phishing is a targeted phishing attack aimed at a specific "
     "person or organization. The attacker researches the victim to craft a "
     "highly convincing, personalized message — often referencing real "
     "projects, colleagues, or events. Because it is targeted, it is harder "
     "to detect than bulk phishing."),
    ("What should I do if I clicked a phishing link?",
     "Disconnect from the network if malware is suspected, change passwords "
     "from a different trusted device starting with the compromised account, "
     "enable or reset MFA, scan the device with updated antivirus, report it "
     "to your IT or security team, and watch for unauthorized activity. Do "
     "not enter credentials again on the suspicious page."),
]

# ------------------------- passwords & auth -------------------------
CATEGORIES["passwords_auth"] = [
    ("What makes a strong password?",
     "A strong password is long (12+ characters), unique per account, and not "
     "guessable from personal information or dictionary words. A memorable "
     "passphrase of several random words works well. Better still, use a "
     "password manager to generate and store random passwords, and secure "
     "important accounts with multi-factor authentication."),
    ("What is multi-factor authentication?",
     "Multi-factor authentication (MFA) requires two or more independent "
     "proofs of identity: something you know (password), something you have "
     "(phone, hardware key), or something you are (fingerprint). Even if a "
     "password leaks, an attacker still needs the second factor. Hardware "
     "security keys resist phishing better than SMS codes."),
    ("Are password managers safe?",
     "Yes, reputable password managers are far safer than reusing passwords "
     "or storing them in browsers or spreadsheets. They encrypt the vault "
     "locally with a strong master password and make unique per-site "
     "passwords practical. The main risks are a weak master password, "
     "phishing for the master password, and compromised devices."),
    ("What is credential stuffing?",
     "Credential stuffing is an attack in which attackers replay username/"
     "password pairs stolen from one breach against other services, "
     "exploiting password reuse. Defenses include unique passwords per "
     "site, MFA, breach-alert services, and on the server side rate "
     "limiting and breached-password screening."),
    ("Should I change my password regularly?",
     "Modern guidance (e.g., NIST SP 800-63B) says forced periodic password "
     "changes often lead to weaker, predictable passwords. Change passwords "
     "when there is evidence of compromise, when a breach affects the "
     "service, or when a password is shared or weak — not on an arbitrary "
     "calendar schedule."),
    ("What is passkeys / passwordless authentication?",
     "Passkeys are phishing-resistant credentials based on public-key "
     "cryptography: the device stores a private key and the service only "
     "holds a public key, so there is no shared secret to steal. "
     "Authentication uses a local unlock (biometric or PIN). They resist "
     "phishing because they are bound to the legitimate site's domain."),
]

# ------------------------- encryption -------------------------
CATEGORIES["encryption"] = [
    ("What is encryption?",
     "Encryption transforms readable data (plaintext) into unreadable "
     "ciphertext using an algorithm and a key, so only holders of the "
     "correct key can reverse it. It protects confidentiality of data at "
     "rest (disk, database) and in transit (TLS for web traffic)."),
    ("What is the difference between symmetric and asymmetric encryption?",
     "Symmetric encryption uses one shared key for both encryption and "
     "decryption — fast, but key distribution is hard (examples: AES, "
     "ChaCha20). Asymmetric encryption uses a public/private key pair — the "
     "public key encrypts or verifies, the private key decrypts or signs "
     "(examples: RSA, ECC). Real systems combine both: asymmetric for key "
     "exchange, symmetric for bulk data."),
    ("What is TLS?",
     "TLS (Transport Layer Security) is the protocol that encrypts and "
     "authenticates internet traffic — it is the 'S' in HTTPS. It provides "
     "confidentiality, integrity, and server authentication using a mix of "
     "asymmetric key exchange and symmetric encryption. TLS 1.2 and 1.3 are "
     "the current versions; older versions are deprecated."),
    ("What is end-to-end encryption?",
     "End-to-end encryption (E2EE) ensures only the communicating endpoints "
     "can read the message content — not the service provider, network, or "
     "any intermediary. The provider transports ciphertext it cannot "
     "decrypt. Signal, WhatsApp, and iMessage use E2EE for messages by "
     "default."),
    ("What is hashing and how is it different from encryption?",
     "Hashing is a one-way function that maps data to a fixed-length "
     "digest; it is not reversible and no key is involved. Encryption is "
     "two-way and reversible with a key. Hashing is used for integrity "
     "verification and password storage; encryption is used when you need "
     "to recover the original data."),
    ("How should passwords be stored in a database?",
     "Passwords should be stored with a slow, salted, memory-hard hash "
     "designed for passwords — such as Argon2id, scrypt, or bcrypt — never "
     "with fast general hashes like MD5 or SHA-256 alone, and never in "
     "plaintext or reversible encryption. Each password gets a unique salt "
     "and a work factor tuned to your hardware."),
    ("What is a certificate authority?",
     "A certificate authority (CA) is a trusted organization that issues "
     "digital certificates binding a domain or identity to a public key. "
     "Browsers and operating systems trust a root CA list; CAs are supposed "
     "to verify control of a domain before issuing certificates for it. "
     "Certificate transparency logs make issuance publicly auditable."),
]

# ------------------------- web security -------------------------
CATEGORIES["web_security"] = [
    ("What is SQL injection?",
     "SQL injection is an attack in which untrusted input is concatenated "
     "into a SQL query, letting the attacker alter the query's meaning — "
     "for example to bypass authentication or dump the database. "
     "Prevention: use parameterized queries or prepared statements, "
     "validate input, apply least-privilege database accounts, and avoid "
     "building SQL by string concatenation."),
    ("What is XSS (cross-site scripting)?",
     "XSS is an injection attack in which attacker-controlled JavaScript "
     "runs in another user's browser, enabling session theft, defacement, "
     "or actions as the victim. Types are reflected, stored, and "
     "DOM-based. Prevention: contextual output encoding, a strict "
     "Content-Security-Policy, HttpOnly cookies, and frameworks that "
     "escape output by default."),
    ("What is CSRF?",
     "Cross-site request forgery (CSRF) tricks a logged-in victim's browser "
     "into sending an unintended authenticated request — for example "
     "transferring funds — because the browser automatically attaches "
     "cookies. Defenses: anti-CSRF tokens, the SameSite cookie attribute, "
     "checking Origin headers, and requiring re-authentication for "
     "sensitive actions."),
    ("What is the OWASP Top 10?",
     "The OWASP Top 10 is an annually-updated awareness list of the most "
     "significant web application security risks. Recent categories include "
     "broken access control, cryptographic failures, injection, insecure "
     "design, security misconfiguration, vulnerable components, and "
     "authentication failures. It is a starting point for appsec programs, "
     "not a complete checklist."),
    ("What is IDOR?",
     "IDOR (Insecure Direct Object Reference) is an access-control flaw "
     "where an application uses predictable identifiers (like /invoice/1234) "
     "without verifying that the requester owns the object. Attackers "
     "change the ID to read others' data. Fix: enforce authorization on "
     "every object access, server-side, not just hide the links."),
    ("What is HTTPS and why should every site use it?",
     "HTTPS is HTTP over TLS. It encrypts traffic so network observers "
     "cannot read or modify it, and authenticates the server so users know "
     "they are talking to the real site. Without HTTPS, credentials, "
     "session cookies, and content are exposed in transit, and modern "
     "browser features (like service workers and HTTP/2) require it."),
    ("What is an SSRF attack?",
     "Server-side request forgery (SSRF) occurs when an application fetches "
     "a URL supplied by the user without restriction, letting attackers "
     "reach internal services — cloud metadata endpoints, admin panels, or "
     "databases — from inside the network. Mitigations: allow-lists for "
     "outbound requests, blocking link-local and private IP ranges, and "
     "network egress controls."),
]

# ------------------------- linux & tools -------------------------
CATEGORIES["linux_tools"] = [
    ("What is Nmap used for?",
     "Nmap is an open-source network scanner used to discover hosts, open "
     "ports, running services, and operating system fingerprints. Security "
     "professionals use it for asset inventory and authorized assessments; "
     "it supports scriptable probes (NSE) for deeper service detection. "
     "Scanning networks you do not own or lack permission to test is "
     "illegal in many jurisdictions."),
    ("What is Wireshark?",
     "Wireshark is an open-source packet analyzer. It captures and displays "
     "network traffic in detail, letting analysts inspect protocols, hunt "
     "malicious traffic, and debug applications. It is passive analysis of "
     "packets you are authorized to observe — it does not inject traffic."),
    ("What is Kali Linux?",
     "Kali Linux is a Debian-based distribution preloaded with hundreds of "
     "security and penetration testing tools (Nmap, Metasploit, Wireshark, "
     "Burp Suite community tooling, and more). It is maintained by "
     "Offensive Security and intended for authorized security work and "
     "learning, not as an everyday desktop OS."),
    ("What is Metasploit?",
     "Metasploit is an open-source exploitation framework with a large "
     "library of exploits, payloads, and post-exploitation modules. "
     "Penetration testers use it to validate vulnerabilities safely and "
     "reproducibly during authorized engagements. It also helps defenders "
     "understand how real exploitation works."),
    ("What are Linux file permissions?",
     "Linux permissions control read, write, and execute access for user, "
     "group, and other (e.g., rwxr-xr-- or 754). They can be inspected with "
     "ls -l and changed with chmod, with ownership via chown and chgrp. "
     "Correct permissions limit what a compromised process or account can "
     "touch. Extended ACLs and capabilities provide finer-grained control."),
    ("What is SSH and how do I secure it?",
     "SSH is a cryptographic protocol for secure remote shell access. To "
     "harden it: use key-based authentication instead of passwords, disable "
     "root login, restrict which users may connect, keep the server "
     "patched, use strong host keys, optionally restrict access with "
     "firewall rules or a bastion host, and consider fail2ban for "
     "brute-force protection."),
    ("What is a Linux log I should monitor for security?",
     "Key Linux security logs include /var/log/auth.log (or "
     "/var/log/secure on RHEL-family) for authentication events, "
     "/var/log/syslog or /var/log/messages for system events, journalctl "
     "output on systemd systems, and web server or application logs. "
     "Forward them to a central SIEM; watch for failed logins, new users, "
     "and privilege escalations."),
]

# ------------------------- incident response -------------------------
CATEGORIES["incident_response"] = [
    ("What are the phases of incident response?",
     "A common model (NIST SP 800-61) has four phases: preparation, "
     "detection and analysis, containment/eradication/recovery, and "
     "post-incident activity (lessons learned). SANS' six-step model adds "
     "identification as its own phase. The goal is a repeatable, practiced "
     "process rather than improvisation during an attack."),
    ("What is a SIEM?",
     "A SIEM (security information and event management) platform collects "
     "logs and events from across an organization, normalizes them, and "
     "correlates them into alerts. Analysts use it to detect attacks, "
     "investigate incidents, and satisfy compliance requirements. Examples "
     "include Splunk, Microsoft Sentinel, and Elastic Security."),
    ("What does a SOC analyst do?",
     "A security operations center (SOC) analyst monitors alerts, "
     "triages suspicious events, investigates potential incidents, "
     "escalates confirmed threats, and contributes detections and "
     "documentation. Tier 1 focuses on triage, Tier 2 on deeper "
     "investigation, Tier 3 on threat hunting and advanced response."),
    ("What is threat hunting?",
     "Threat hunting is proactive, hypothesis-driven searching through "
     "telemetry for attackers that automated detections missed. Hunters "
     "start from an indicator, a technique (often mapped to MITRE "
     "ATT&CK), or an anomaly, then investigate with SIEM queries, EDR "
     "data, and endpoint artifacts. Findings feed back into new detections."),
    ("What is MITRE ATT&CK?",
     "MITRE ATT&CK is a public knowledge base mapping real-world attacker "
     "tactics (the 'why') and techniques (the 'how') across platforms. "
     "Defenders use it to map detections, prioritize coverage gaps, plan "
     "adversary emulation, and communicate threat behavior in a common "
     "language."),
    ("What should be in an incident response plan?",
     "A solid plan defines roles and contact lists, how incidents are "
     "declared and escalated, containment and evidence-handling "
     "procedures, communication templates (internal, customers, "
     "regulators), legal and HR interfaces, backup and recovery steps, "
     "and a testing/rehearsal schedule. It should be written before an "
     "incident and exercised regularly."),
    ("What is digital forensics?",
     "Digital forensics is the collection, preservation, and analysis of "
     "digital evidence in a way that maintains its integrity and "
     "admissibility. Practitioners work with disk images, memory captures, "
     "and logs, following strict chain-of-custody procedures. Tools "
     "include Autopsy, Volatility, and commercial forensic suites."),
]

# ------------------------- privacy & risk -------------------------
CATEGORIES["privacy_risk"] = [
    ("What is the CIA triad?",
     "The CIA triad is the core model of information security: "
     "confidentiality (only authorized parties can read data), integrity "
     "(data is not altered improperly), and availability (systems and data "
     "are accessible when needed). Most security controls map to at least "
     "one of these three goals."),
    ("What is risk management in cybersecurity?",
     "Security risk management identifies assets and threats, assesses "
     "likelihood and impact, and decides how to treat each risk: mitigate "
     "with controls, transfer (insurance), accept formally, or avoid by "
     "changing the activity. It is continuous — risks are re-assessed as "
     "technology and threats change — and ties security spending to "
     "business impact."),
    ("What is data privacy vs data security?",
     "Data security protects data from unauthorized access or tampering — "
     "it is about controls. Data privacy concerns the rights and "
     "expectations around how personal data is collected, used, and "
     "shared — it is about governance and consent. You can have strong "
     "security and still violate privacy by misusing data."),
    ("What is GDPR?",
     "GDPR (General Data Protection Regulation) is the European Union's "
     "comprehensive data protection law. It grants individuals rights over "
     "their personal data (access, erasure, portability), requires lawful "
     "bases and privacy-by-design for processing, mandates breach "
     "notification, and carries significant fines. It applies to any "
     "organization processing EU residents' data, regardless of location."),
    ("What is security awareness training?",
     "Security awareness training teaches employees to recognize and "
     "respond to threats — phishing, social engineering, safe data "
     "handling, password hygiene, and reporting procedures. Effective "
     "programs are continuous, scenario-based, measured (e.g., simulated "
     "phishing click rates), and backed by a blame-free reporting culture."),
    ("What is defense in depth?",
     "Defense in depth layers multiple independent controls so a single "
     "failure does not cause a breach: for example MFA plus endpoint "
     "protection plus network segmentation plus monitoring plus backups. "
     "Assume any one layer can fail and design so the next layer catches "
     "the attack."),
    ("What is a vulnerability vs an exploit vs a threat?",
     "A vulnerability is a weakness in a system. An exploit is code or a "
     "technique that takes advantage of that weakness. A threat is any "
     "actor or event that could cause harm — including but not limited to "
     "those exploiting vulnerabilities. Risk combines the likelihood a "
     "threat exploits a vulnerability with the resulting impact."),
    ("What is cyber hygiene?",
     "Cyber hygiene is the routine, foundational maintenance that prevents "
     "most common attacks: keeping systems patched, using unique passwords "
     "with MFA, backing up data, removing unused accounts and software, "
     "encrypting laptops and phones, and reviewing access regularly. It is "
     "the security equivalent of brushing your teeth."),
]


def validate_sample(instruction, output):
    if not instruction.strip() or not output.strip():
        return False
    if len(instruction) < 5 or len(output) < 40:
        return False
    if output.count(output[:60]) > 1:
        return False
    return True


def generate(num_per_category, seed=42, categories=None):
    rng = random.Random(seed)
    samples = []
    for cat, pairs in CATEGORIES.items():
        if categories and cat not in categories:
            continue
        if num_per_category <= len(pairs):
            chosen = pairs
        else:
            chosen = pairs + [
                rng.choice(pairs) for _ in range(num_per_category - len(pairs))
            ]
        # dedupe within category
        seen = set()
        for q, a in chosen:
            key = (q, a)
            if key in seen:
                continue
            seen.add(key)
            if validate_sample(q, a):
                samples.append({
                    "instruction": q.strip(),
                    "input": "",
                    "output": a.strip(),
                    "category": cat,
                })
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-per-category", type=int, default=4,
                        help="Samples per category (caps at pool size by default)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-duplicates", action="store_true",
                        help="Allow repeating samples beyond pool size")
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    n = args.num_per_category if args.allow_duplicates else min(
        args.num_per_category, max(len(v) for v in CATEGORIES.values()))

    samples = generate(n, args.seed)

    # global dedupe
    seen = set()
    final = []
    for s in samples:
        key = s["instruction"].lower()
        if key in seen:
            continue
        seen.add(key)
        final.append(s)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for s in final:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Wrote {len(final)} samples to {args.output}")
    cats = {}
    for s in final:
        cats[s["category"]] = cats.get(s["category"], 0) + 1
    for c in sorted(cats):
        print(f"  {c}: {cats[c]}")


if __name__ == "__main__":
    main()
