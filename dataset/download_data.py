"""
Dataset Helper & Generator Script.

This script ensures that Fake.csv and True.csv exist in the dataset/ directory.
If user CSVs are missing, it generates a comprehensive synthetic dataset adhering to
the Kaggle/ISOT Fake and Real News dataset format (title, text, subject, date).
"""

import os
import sys
import pandas as pd
import numpy as np

# Reconfigure stdout to UTF-8 for Windows consoles with unicode paths
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
FAKE_CSV_PATH = os.path.join(DATASET_DIR, "Fake.csv")
TRUE_CSV_PATH = os.path.join(DATASET_DIR, "True.csv")


def generate_sample_datasets():
    """Generates realistic sample Fake.csv and True.csv files for immediate testing."""

    print("Generating comprehensive dataset samples for training and validation...")

    # Real news templates and components
    real_titles = [
        "Federal Reserve Announces Interest Rate Decision Following Policy Meeting",
        "European Union Approves Landmark Artificial Intelligence Regulation Framework",
        "NASA Spacecraft Collects Samples From Distant Asteroid Surface",
        "Global Climate Summit Concludes With Renewable Energy Commitments",
        "Tech Giants Report Quarterly Earnings Ahead of Wall Street Expectations",
        "Bipartisan Bill Signed into Law to Upgrade Infrastructure and Roads",
        "World Health Organization Issues Guidance on Seasonal Virus Preparedness",
        "Major Automobile Manufacturer Unveils New Electric Vehicle Fleet",
        "Supreme Court Issues Rulings on High-Profile Regulatory Rights Case",
        "Economic Growth Exceeds Analyst Expectations in Third Quarter Report",
        "International Trade Negotiations Reach Consensus on Digital Tariffs",
        "Archaeologists Unearth Ancient Historical Settlement in Mediterranean",
        "Central Banks Coordinate to Stabilize International Currency Markets",
        "Scientists Publish Breakthrough Findings in Renewable Battery Storage",
        "United Nations Security Council Passes Resolution on Regional Peacekeeping",
        "Senate Confirms Federal Judge Candidates in Bipartisan Session",
        "Labor Department Reports Drop in Unemployment Rate Across Sectors",
        "Medical Researchers Complete Clinical Trial for Target Cancer Therapy",
        "G7 Leaders Meet to Discuss Cybersecurity and Global Supply Chains",
        "Energy Department Grants Funding for Offshore Wind Farm Development"
    ]

    real_texts = [
        "WASHINGTON (Reuters) - The Federal Reserve announced today its latest monetary policy decision following a two-day meeting. Financial officials indicated that interest rate adjustments will remain dependent on incoming inflation data and labor market indicators. Economists noted that market indices responded with moderate gains following the official release statement.",
        "BRUSSELS (Reuters) - European lawmakers voted overwhelmingly to adopt a comprehensive legislative framework governing artificial intelligence applications. The new rules mandate transparency, risk management benchmarks, and strict safeguards for high-risk deployments. Industry leaders commended the decision as a standard-setting initiative.",
        "CAPE CANAVERAL (Reuters) - NASA mission controllers confirmed successful retrieval of surface samples from a carbon-rich asteroid. The robotic spacecraft navigated autonomous thrusters to secure physical specimens for analysis. Scientists expect the findings to shed light on early solar system formation.",
        "PARIS (Reuters) - Representatives from over 190 nations concluded annual climate negotiations with a ratified agreement prioritizing renewable energy expansion. Delegations agreed to double solar and wind infrastructure investments over the coming decade while providing financial aid to developing regions.",
        "NEW YORK (Reuters) - Technology sector earnings topped consensus estimates as Cloud computing revenue accelerated across enterprise segments. Corporate executives highlighted expanding margins during conference calls with investors today.",
        "WASHINGTON (Reuters) - President signed legislation allocating federal grants toward highway renovation, bridge repairs, and clean transit networks. Both congressional parties applauded the legislation as a vital infrastructure modernization milestone.",
        "GENEVA (Reuters) - The World Health Organization issued updated global public health guidelines emphasizing preventive healthcare and early diagnostic monitoring across healthcare systems worldwide.",
        "DETROIT (Reuters) - Auto manufacturers announced next-generation battery architectures offering extended driving ranges and faster charging capability. Production lines are scheduled to begin output next year.",
        "WASHINGTON (Reuters) - The Supreme Court handed down a decision addressing regulatory authority standards under administrative law. The majority opinion noted the necessity of adhering strictly to statutory provisions.",
        "CHICAGO (Reuters) - Department of Commerce economic figures showed gross domestic product expanded at an annual rate exceeding initial forecasts driven by consumer spending and manufacturing strength."
    ]

    # Fake news templates and components
    fake_titles = [
        "SHOCKING BREAKING: Secret Government Machine Controls Weather Across Globe!",
        "CONFIRMED: Aliens Have Infiltrated High-Level Political Cabinet Meetings!",
        "MUST SEE: Drinking Miracle Juice Cures All Known Diseases Instantly!",
        "BOMBSHELL EXPOSÉ: Hidden Underground City Discovered Beneath Ocean Floor!",
        "LEAKED DOCUMENTS: Banking Syndicate Plan to Replace Money with Digital Microchips!",
        "UNBELIEVABLE: Celebrity Reveals Mind Control Script Used by Mainstream Media!",
        "PROOF: Time Traveler from 2085 Predicts Tomorrow's Lottery Numbers!",
        "CELEBRITY EXPOSED: Secret Hollywood Portal Grants Eternal Youth to Elites!",
        "ALERT: Suppressed Technology Allows Cars to Run Unlimited Miles on Plain Water!",
        "SCANDAL: Secret Society Controls World Elections Using Psychic Frequencies!",
        "BOMBSHELL: Ancient Pyramids Were Secret Spaceports Built for Intergalactic Ships!",
        "WARNING: Microwave Ovens Broadcast Secret Surveillance Signals to Headquarters!",
        "EXCLUSIVE: Missing Island Discovered Hidden Inside Huge Cloud Vortex!",
        "BREAKING: Scientists Confirm Moon is Hollow Spacecraft Guarded by Robots!",
        "SHOCKING TRUTH: Hidden Code in Ancient Painting Predicts End of World Next Week!",
        "LEAKED: Shadow Organization Synthetic Humans Replacing Government Officials!",
        "BOMBSHELL CLAIM: Invisible Energy Shield Surrounds Lost Continent of Atlantis!",
        "REVEALED: Secret Elixir Found in Antarctic Cave Grants Telepathic Abilities!",
        "EXPOSED: Global Conspiracy Hides Giant Flying Dragons Living Inside Earth Core!",
        "MUST READ: Instant Teleportation Devices Already Sold in Secret Dark Markets!"
    ]

    fake_texts = [
        "In a mind-blowing revelation, anonymous whistleblowers claim that secret military installations possess top-secret frequency cannons capable of manipulating global weather patterns at will. Mainstream media refuses to report this terrifying truth! Share this video before it gets deleted by the censorship authorities!",
        "Shocking photos leaked on deep web forums allegedly prove that extraterrestrial entities have secretly taken over major government positions! Insider sources reveal exotic technology is being tested in hidden underground bunkers without public knowledge. Wake up people!",
        "A rogue scientist who was banned by big pharmaceutical companies claims that drinking a mixture of cucumber juice and special crystals instantly cures every illness known to mankind! Doctors hate this simple trick! Click here to order your secret bottle now!",
        "Unbelievable documents uncovered from a secret vault show an entire mega-city thriving miles below the ocean floor! Authorities are scrambling to cover up satellite footage showing giant underwater domes and alien tech.",
        "Leaked files reveal a shadowy cabal of global elites planning to dismantle all physical currency by next month and force every citizen to receive a biochip implant! Financial experts urge everyone to stock up on canned food immediately!",
        "A former Hollywood insider just broke silence and exposed the secret scripts given to TV anchors to control human minds through subtle light flashes. Watch this unedited footage before it gets banned across all platforms!",
        "Eyewitnesses in downtown area claim a man wearing futuristic clothing appeared out of nowhere and accurately predicted local events before vanishing into thin air! Government agents were spotted confiscating security tapes.",
        "Exclusive report reveals suppressed patents from 1920 that prove internal combustion engines can easily run on pure tap water! Big energy conglomerates paid billions to bury this technology forever!",
        "Scientists who refused to sign confidentiality agreements claim that microwave appliances are transmitting secret spy signals directly into citizen homes! Unplug your electronics now!",
        "Secret ancient texts decoded by independent researchers prove that massive pyramids were built as refueling docks for alien starships! Official historians are desperately trying to censor the evidence!"
    ]

    np.random.seed(42)

    # Build 250 rows of real news
    real_data = []
    for i in range(250):
        t = real_titles[i % len(real_titles)] + f" (Report #{i+1})"
        b = real_texts[i % len(real_texts)] + f" Additional verified analysis confirmed baseline stability across indicators in quarterly update sample #{i+1}."
        subj = "politics" if i % 2 == 0 else "worldnews"
        date = f"December {1 + (i % 28)}, 2023"
        real_data.append({"title": t, "text": b, "subject": subj, "date": date})

    # Build 250 rows of fake news
    fake_data = []
    for i in range(250):
        t = fake_titles[i % len(fake_titles)] + f" !!! [Must Watch #{i+1}]"
        b = fake_texts[i % len(fake_texts)] + f" Conspiracy whistleblowers confirm this secret evidence in private report #{i+1}!"
        subj = "News" if i % 2 == 0 else "Government News"
        date = f"December {1 + (i % 28)}, 2023"
        fake_data.append({"title": t, "text": b, "subject": subj, "date": date})

    df_real = pd.DataFrame(real_data)
    df_fake = pd.DataFrame(fake_data)

    df_real.to_csv(TRUE_CSV_PATH, index=False)
    df_fake.to_csv(FAKE_CSV_PATH, index=False)

    print(f"Successfully generated dataset at:\n  - {TRUE_CSV_PATH}\n  - {FAKE_CSV_PATH}")


def ensure_datasets_exist():
    """Checks dataset directory and creates sample files if necessary."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    fake_exists = os.path.exists(FAKE_CSV_PATH)
    true_exists = os.path.exists(TRUE_CSV_PATH)

    if not fake_exists or not true_exists:
        print("Dataset files (Fake.csv / True.csv) not detected in dataset/ directory.")
        generate_sample_datasets()
    else:
        print(f"Dataset files detected:\n  - {FAKE_CSV_PATH}\n  - {TRUE_CSV_PATH}")


if __name__ == "__main__":
    ensure_datasets_exist()
