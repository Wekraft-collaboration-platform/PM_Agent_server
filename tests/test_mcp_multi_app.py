"""
test_mcp_multi_app.py — Complete End-to-End Multi-App MCP Sub-Agent Test.

Query: "get me all issues and 3 most critical issues from sentry, plus jira tasks and vercel deployment status of wekraft"
Tested integrations: Sentry, Vercel, Jira in parallel.
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure 'src' is in pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

load_dotenv()

from langchain_openai import ChatOpenAI
from app.agents.tools.mcp_client import (
    smart_prune_connectors,
    _execute_single_app_worker,
    WorkerResult,
)


def decrypt_field_sync(data: dict) -> str:
    """Decrypts AES-256-GCM token from Convex credentials object."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    import binascii

    key_secret = "wekraft_default_sec_key_32bytes!"
    raw_key = key_secret.ljust(32, "0")[:32].encode("utf-8")
    aesgcm = AESGCM(raw_key)

    iv = binascii.unhexlify(data["iv"])
    ct = binascii.unhexlify(data["ciphertext"])
    tag = binascii.unhexlify(data["tag"])
    return aesgcm.decrypt(iv, ct + tag, None).decode("utf-8")


# Fresh Decrypted Credentials from Convex
SENTRY_CRED = {
    "ciphertext": "0f70d21b65ccad616b6a41f8c629bccce211a1475df7136faef19653b542832f6df972e39d7fcb30ee49825d954e029160f87e3df12e961c56",
    "iv": "a2053cccfe5e72c987a9a88b",
    "tag": "7cb5500f5c57a918d02e300f4d76f3f9",
}
VERCEL_CRED = {
    "ciphertext": "4a4f55b9aa03b5ab4914557e82ce84b3b11eebc3f4591812ef11ccb325f295d108a733110457596cd3a376895fb1772cff99e0e4c4effe7e15695a97",
    "iv": "cdb76c72412e3421f4ae79c6",
    "tag": "61b5bcde8c36b17e50e93506700fc402",
}
JIRA_CRED = {
    "ciphertext": "f032dac12083144817d3f3ab8b780bd72f138ba08dc75f5edcd9d29e08c3f1cf83f7ab509003780f6e1244952d56ad331de82d4dde480627301f610ed8d1256b91d794187cdba68b2684ead1cea69254b06e89108a6cfc9d1a6ed747bcbe268a001f773d294ac473a5ad29b8e6193a51e759a65dfa4e7159b96be1302f78824a70087fc67c5179de7d1cef84269ea10ac17e8143e6fa11c168bab937f348ad37d7189833da7c68ae02882fe0d99753dc6c0e69b2fdb0a48b48532776623fa20774f23cae107dea46b9b873f6f22408a4e50c6dad785e7aae14afa71bcf07b7f3ce400113ec043cbbeb6024eb80b40ed00ffb3158419f3ce7b107883ba93a1afda6a5f6281b7b4276fe893bedec1910018053178f84f97cff49bc058ee303c50c9ba07de315f481e37544b6dfa327c2a4649b74d8510b62dab232239861d6fd9ffa3a845136654aa0ff7a97e26fa46bda6f3651462ede8ee405f555a094a48a1eaa8379c62555940f5d6ec2bf14d57d04ac8204ab6bb470bae0e8e8813ab7c290307f450b721a8af2425a05c0e88574c3427484b723e617f52441c827f023ed727c697997e1267a38b3a4e30756c9df9264d8e428da35d56e0c1635d845f5c1bda8e8cfafe39fcc8b9af5f78cd4815f1b129671c2c7773a481e2cc5744294cfc13332c79a9447b208c0a6f0fcd71f50feb1b5e203d98f18059331463f137d0f387e4794f074c92376aad952ad39f93b91b5ffe00eb9333b7579556b83b47235e71543dd4259802e278d6dfc1252f764fb685104d5497572b98343f2aa8851faccb56233ec05477bba88ffcff871879721327689a9cf47b8920b84a2572057bb7d630138edd33f07fed6619b8a8a2dc5dce2a3ba87a9191b9358a67a1f404ef829779e5f89f9c5ee51b6cd7e1612e416e263c4d2d7891794f726febb04674fdcac87f96fc185f983545893f3d28598ba9f054126f7e898ca1fe35a58d452bc775035183358c9553e430f6a57af102230f8f7b88bbb213e806d8ebb1ab0d48e44660169e88c2dfcd258e2c0f3ccc1dd376dcd476ce04a4d6dc9649f65192ba956c867d0fb477f25cb3b59720adaa76e7711d9c1ba32b49fa1edf9f5bccc37d598ee5695ea2d259c362ca907bc2ace143bac634bfcda917a8ebafadd881e39b7495e3d201fa74674ccc9b02c1402d68f93968b8118bb54a8fc7e28ea53bc455fc72f3c7b72b9505ede6861484ef53402512c598d01093a0116352e657a47d038827a11eef97da8f2f76edb07746bcad6a9e34589b4fd78746c9f611ef015b2186bc10ff969ea1820d6233a0cb6a45a02bfad3cae67932c22eb4cc27999235d68b3253a286b21288006c2b4d9ad16bba2447e24a5546a0275fff6ef5172ec6b24882b266ddbfd59090a44c8a8ca15651b8e891e6d61c9500bfa09db83efc6db120ef3583d557629f176f49cc5817c1bae4a62855087eb6fdd94fd699833ce10d3c4ff5922a21948a4f4856ddcea10f86e47ae88b855e59cf81ecc35f08f1cbb44f2af4745689c785d073dcb3f2617a4054f98159cb829428024c5fbc8a2e3e3d7f8837b0a56bb1d5e90e4876666d1c96fb8051758460fa18cf66c4d6471ee217aa32e9b9738f97f9c38273f427c3edba39f0408e8eefcb4aed7d7f31bc7b84bf0d130f894ef7c412c6cc7a62d6d5669eac91640765d3584c26c3589879ca5ab849d786c30ac7c8ca2aa02c317e8f99ed873b4f72cdaa3f41b91cf4e31ca230045f218c48c926d08668cef2f4a7c5b1aa42d41b8bd9a3755338b5ec67d991c190a65efbd85cbb207e6eea5e4438ecdca2974d3b9a7c6c46626c16b4a0a845e1b9e8fef5af3728c047bfa238b0313b1677ffb803784924b738ba9c7f86bc861f2689c2d1d16e302c71464ac47ce56ac99a271aad1b9a9f55bf8a7dd029b24af5d5ce2ef5f46e46858942d8175ca4e727258fc6058975759134c0850f524e2576df5a7a8cc046894d075429905baa418a0708cd9d3db528efbc49e1159f89f543a4556bf53620b878c73617caff1b551ef79e410b02d2a9bde7708d1627746faaf19b52cfc3f622a72a107f02828572840edc4e7e0f77137c551ed44903b0be7860860f0f7dbb4dfa6101f40f8f5d51dcd6e8b5638dea2ee04a7aa7b687a3f39235fc25596cebf5b405e7aa40e77136afb94895f57d39d1b7d1a808ab091c7ad4338d1024401fc13305333d7388c04a7c5dd20b60ceebdf8f3aa8cc03b5c0decee26ff6f19563983a986b7ab0fc4d9abdf21f0b0fe60a4d8ddba613511fd28a4feb6cf26f848e8157929c97b769c9eb4731bb28ac391812a2f3609600617473413055804f5da56c73fd0847191a3327dccb3863108754a43a12ce1f3b3f6dc216be2b87f26a82aa5c7990fcc931af1d41057277d1c6e7070d0e7f75a880b7c477b5cab8432158bdc1a2f9243ccea32f8157ddcb6fafabe997950e76f8b59ab77bc3ea5cdc4180b87e7c2d790dc3b01b90f1e78c1162caca3f66afecc7adf945b47e62e658c964c54c4290b1f7168be5cafbe61061ce66abb2d36be267b7aba296cb2084117a5a2a509f89c9386442d68786d636c01b86d0a025d66e445b79d2678e495011d47897c414a695bbee96b0ac31baa1790384fa231cfeceaac9533f1bc82e0cb105674322e2b45510c6c12d6c9b6f992292580593c14faccdcf42aa0a41fb2dc7dc3afb7ab1010a6cccfbed9257649080daa0dd537b453b77fedc004360f741d332b5470b6c806ccfcf55e1c8581a95839db3f19019bf56b6afdee7d4fd169967e16522c0209e79861cd10cd79f26683de4411fb970f6d479025e67161f94499ee62910b237635ecfe38ca05ad78b4c1f3fdb00f43c4a3f24b8a83a4023bf1aef43c79c872325861abc31549ca5136054cde2110576408f53e3d9d7c4f2fbd01619fe98b665245c51892b22c0d48fbb4840011fa18432fb2cfcda69dd36e8bdb0a1852cd3897e445503be4c5e0fbe50ea170bcdc1ffd957ac003a4d193cb3191f100ea0e1b3f8b5fa853a5c82e5e9169855547f7fd38b904ea6cda6def2a46331bfb35094807f0922d4f4b5fda2f777174764f8c630a1f1e2b8db93a6a0554f0fbfdbb9d0872219c67ecb692fcaf4cedaf19d49bb476e50b2c367dd1435ef457ef7b3fb66bcd8486321545666439c64816015a8aa635ff1dd9a6a3cfc8b245d8412a49bd7002ee87c8433bcdbef8657e2309bb416eb676774ed8246891c99d7c52188acbc624153ef789dc8e07bf7507a2d5feed2eb51bffe4c79c1ff71a80233d59a8d9eaf7e27234a4b2df97ab812d986f824ac514b6e72b487b87848f1ada492957fb84efdc26bd1397b457b998eddff3fe9c613356fa48b06cb44ddd2d9f95a99f981da7e381bf636262f4df0fc05a883b4ed1257107b75d4ba6bd8710a5f103dd5a47fc59e9b0362a37e6640c69bec2f59b6cc14f3b5e06b13365b2f376943bed906031e78a4dabbd1c90592553343fcb774a2b586d48233ac3fedd9cb5894c3af98485af4e645d308077b2788759d32e2aa0df506bea79bc99aba3e9b9a68cab374103e82f04d4bd239a5c089de3567edf113796265d97d3f79b5af8984a09792a831a47cc1e29ab74d61ebec7eba84b53761d39ca11546876dd2d7714d7570306e355157d6237e9f456e6de92838c7197fa6051dcfc8a93a6379983a33cc30ecf5f7e88bcf5f7ac1efdc7a15192e1db6b9e33802aad3b6a19eb048036e80e3de50504a24becdbff5a4a811567e9c85ee3836e8a3621fe5c8a5e9a6b9f2b7f31c4c1c87c02c8ec98c0fc65c9a688a37fdfd7566b6cc4d0cc3d11d943caf029f11a082bb56b3e956dda14e0b69e88015689d8f06ec5c7e852927a28f1d0fd81bfc99a61fd228c9796c0cce59fcad4bccaabe9661c1fac6fbdc3823667b0dfea9ef8cd8701ded44794bd11a44a243792df5cfd9d9967613cacbaa782eea44eb1386b52be3cc8e3f107215144a5689c7d8ff98ca012f5e1cae5286054d43bd1796b30352c278e4a25d7bf40088141fc08d01d4a0d918020537477551670418cd91af78e572626f96b9b6f024a05d341bf9115b392ce5038c89f1b6cbd23e1f47a801368be5cb6ff3818c3897855d934022abe7276ff710681c808407a9f11197bcf87c6192b953cef3428017907512ffdebc0b637b2017d35a3c0d9923dea3f22e59988e158b033f44e8eebea7a56a4d1b5634abc0f587448b6ff3fc89348cd5ab7c77ec18540d364931ac223ea62eccc7d76bb9154d1d00a90952d82478f1f20a71affe4effb622fcc0c99bbdcd1c25cd2f398d054ba6476c88b115df5d5a8dbf72680a2c6ef7faa6a5465f68a7cf2983a98795e34ceb2217141d409c4d2c27b6d211162f20df0c9d1e1336bd9712903755f3d0fba4fc26a192c030df4bdb5a97efe2130caaf2f4027b9901321907bbfc49bd088e1c1b560913c0823ff4b7e753a08e479568dd9312f72b600171e5b5ceac74b870b8350d2dcf7e2021449525c924dadf6127c2de86064ce136452637d4f60b7354e9a54add0ef84a2bf5c64882808b786f2d132b19dc6332fbaa4d9c9d554dbae8851297fdc32344851fdc2c1def36dfc13a40495e9412d6cb9b4408fb6ad890367204e16dabb5fb35434141afe7c591e551adcce01a2d371d277b7f20710ab4f236e0b5c93ece337f73ba79edfe3baa891c380e513523076a103cde682da03f6f277cf82eb16647564554310ae1e816bd3578ac1dff7c5af88da50b2ac51918843b16f1369dfdad6044120cbf8e4b6eed184f335caed8a70135abfa5b5e17b6ad858a311962ea9c85955ef89b3088d9b3060b8f25ad934874912193a72a332ed7dfc11cfd65d5591bcb2f2f6b92683da993326d46856aad415bca225c392019f327a8fafde5b2f8fd59a5279ebce530641a3cb20ac9cfc4be8dfb6113f79b823dc79effce2846d6e85f55534ad55ba20767113ee2bf38114fc62f41b0c38197401ff79750efe37c0100567ece9222abc9544b88e99f9dc37a147b233f6c04d7634631fc0b9b79d6720e5781c39d7f569f956b62df00c53d6365161ae707597477a3a29bef69e37c24063ae96b11f6ed2ec98a279ad8a68953ca5a704eae36fa1a12b4d32deb1ab6c4f010831c2dd694a4f1af9c56da8310cca619ee706145190dd866b63f436f636330a9a6a591caf32f002c8ff07b058386ff3fb34b2f9213670af796cdca00d9637885604012dd9192c4a54037ec5cafe6fc3ffb59837ce11ac208fb88cf90aeaf5ab3bd9d6388b19f509a0d2c1d8efcf8dc7e45f44edede623f3a1c70654ffd7f74337d2aa1c6cd14160aed918efe81ed763d6e343f306e2f3081abf9108910252118fbb84f709bc9858de3f3444b5979c131087effa70c8e9fc6ea3f72d3ed53f7457dbaf7cc362d2ff4318614b377c21585cecffa668f2028fc245043e6ba9bbaabbd5cee43dd96239525bb91997ea32bd78a7a8cd80c98bbac1e81e82621e85ffef133738c615656a74579495ab5354c6eaa43b42827cb00ddd795093d735b1290027bcd9d1710748fdfbe5673037bcdf267b10",
    "iv": "20f2b36611cf61ae5d7f413b",
    "tag": "d4fabadf03a4d9299140e1ae0da1359a",
}


async def run_live_test():
    user_query = "get me all issues and 3 most critical issues from sentry, plus jira tasks and vercel deployment status of wekraft"
    print("=" * 80)
    print(f"TEST QUERY: '{user_query}'")
    print("=" * 80)

    # 1. Decrypt credentials
    sentry_token = decrypt_field_sync(SENTRY_CRED)
    vercel_token = decrypt_field_sync(VERCEL_CRED)
    jira_token = decrypt_field_sync(JIRA_CRED)

    print(f"\n[1. CREDENTIALS STATUS]")
    print(f"  ✓ Sentry Token: {sentry_token[:15]}...")
    print(f"  ✓ Vercel Token: {vercel_token[:15]}...")
    print(f"  ✓ Jira Token:   {jira_token[:25]}... (Length: {len(jira_token)})")

    # 2. Setup mock active connections
    connections = [
        {
            "connectorId": "sentry",
            "accessToken": sentry_token,
            "metadata": {"displayName": "Sentry Monitoring", "mcpUrl": "https://mcp.sentry.dev/mcp"},
        },
        {
            "connectorId": "vercel",
            "accessToken": vercel_token,
            "metadata": {"displayName": "Vercel Deployments", "mcpUrl": "https://mcp.vercel.com"},
        },
        {
            "connectorId": "jira",
            "accessToken": jira_token,
            "metadata": {"displayName": "Atlassian Jira", "mcpUrl": "https://mcp.atlassian.com/v2/mcp"},
        },
        {
            "connectorId": "linear",
            "accessToken": "dummy_linear",
            "metadata": {"displayName": "Linear Workspace"},
        },
    ]

    # 3. Test Routing (Smart Prune)
    matched = smart_prune_connectors(user_query, connections)
    matched_ids = [c["connectorId"] for c in matched]
    print(f"\n[2. ROUTING VERIFICATION]")
    print(f"  Matched connectors: {matched_ids}")
    assert "sentry" in matched_ids, "Sentry must be matched"
    assert "vercel" in matched_ids, "Vercel must be matched"
    assert "jira" in matched_ids, "Jira must be matched"
    assert "linear" not in matched_ids, "Linear should be excluded"
    print("  ✓ Routing isolated Sentry, Vercel, and Jira in parallel!")

    # 4. Initialize LLM
    openai_key = os.getenv("OPENAI_API_KEY", "")
    llm = ChatOpenAI(model="gpt-4.1-mini", openai_api_key=openai_key, temperature=0.0)

    # 5. FAN-OUT: Run all 3 sub-agents concurrently in parallel
    print(f"\n[3. PARALLEL FAN-OUT SUB-WORKERS]")
    print(f"  Launching Sentry, Vercel, and Jira workers in parallel via asyncio.gather()...")

    start_time = asyncio.get_event_loop().time()
    tasks = [
        _execute_single_app_worker(conn, user_query, llm, timeout_s=45.0)
        for conn in matched
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    duration = asyncio.get_event_loop().time() - start_time

    print(f"\n[4. EXECUTION SUMMARY: Completed in {duration:.2f}s]")
    print("=" * 80)

    summary_sections = []
    structured_data = {}

    for res in results:
        if isinstance(res, Exception):
            print(f"Worker Error: {res}")
            continue

        print(f"\n--- {res.connector.upper()} WORKER ---")
        print(f"  Status: {'✅ OK' if res.ok else '⚠️ FAILED'}")
        print(f"  Tools Executed: {res.tools_executed}")
        print(f"  Resolved Context: {res.resolved_context}")
        print(f"  Structured Items Count: {len(res.data)}")
        if res.error:
            print(f"  Error: {res.error}")
        if res.summary:
            print(f"  Summary:\n{res.summary}")
            summary_sections.append(f"### {res.connector} Findings:\n{res.summary}")
        elif res.data:
            summary_sections.append(f"### {res.connector} Findings:\n{json.dumps(res.data[:3], indent=2)}")

        if res.ok:
            structured_data[res.connector] = res.data

    print("\n" + "=" * 80)
    print("FINAL CONSOLIDATED OUTPUT FOR KAYA SYNTHESIZER NODE:")
    print("=" * 80)
    print("\n\n".join(summary_sections))
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_live_test())
