"""Demo catalogue: 35 consumer electronics products across 8 categories, plus 3 users.

Real products, because a reviewer can judge instantly whether recommending the Bose
QuietComfort Ultra to someone who lingered on the Sony WH-1000XM5 is sensible — that
judgement is much harder to make about an unfamiliar catalogue.

Descriptions are written rather than templated: retrieval quality depends on them, and
a catalogue of "A good product" embeds into mush. Every fact here is also the *only*
thing the agent is allowed to claim about a product, so the spec line matters.

    uv run python -m app.seed          # idempotent: safe to re-run
    uv run python -m app.seed --reset  # wipe first
"""

import argparse
import logging

from sqlalchemy import select

from app.db import init_db, session_scope
from app.logging_conf import configure_logging
from app.models import Base, Product, User, UserProfile
from app.schemas import ProductIn
from app.security import hash_password
from app.services import outbox
from app.services.catalog import create_product

log = logging.getLogger(__name__)

# title, brand, category, tier, price$, rating, tags, spec, description
PRODUCTS = [
    # ---- audio -------------------------------------------------------------
    ("Sony WH-1000XM5 Wireless Headphones", "Sony", "audio", "flagship", 399, 4.8,
     ["noise-cancelling", "over-ear", "wireless", "travel"],
     "30h battery · adaptive ANC · multipoint Bluetooth 5.2 · 250g",
     "Eight microphones and two processors cancel low-frequency cabin noise well enough that long flights stop being tiring. Light clamp force for all-day wear, and the folding case survives being crushed into a bag."),
    ("Bose QuietComfort Ultra Headphones", "Bose", "audio", "flagship", 429, 4.7,
     ["noise-cancelling", "over-ear", "wireless", "spatial-audio"],
     "24h battery · immersive spatial audio · Bluetooth 5.3 · 250g",
     "The strongest noise cancellation on the market paired with a head-tracked spatial mode that places the sound in front of you rather than inside your skull. Comfortable enough to forget you are wearing them."),
    ("Apple AirPods Pro 2 (USB-C)", "Apple", "audio", "mid", 249, 4.7,
     ["earbuds", "noise-cancelling", "wireless", "apple"],
     "6h buds / 30h case · adaptive audio · USB-C · IP54",
     "In-ear cancellation that adapts to your surroundings in real time, plus conversation awareness that ducks the music when you start speaking. Seamless if the rest of your devices are Apple."),
    ("Sennheiser HD 660S2 Open-Back Headphones", "Sennheiser", "audio", "flagship", 599, 4.6,
     ["open-back", "wired", "audiophile", "studio"],
     "300 ohm · open-back · detachable cable · 6.3mm and 4.4mm",
     "Open-back reference headphones for listening at a desk, not on a train. Deep, textured bass extension and an unhurried midrange, but they leak sound in both directions and want a proper amplifier."),
    ("Anker Soundcore Q30 Headphones", "Anker", "audio", "entry", 79, 4.4,
     ["noise-cancelling", "over-ear", "wireless", "budget"],
     "40h battery with ANC · hybrid ANC · multipoint · 260g",
     "The sensible budget pick. Noise cancellation good enough for an open-plan office and battery life that outlasts anything costing four times as much."),
    ("Sonos Era 300 Smart Speaker", "Sonos", "audio", "flagship", 449, 4.5,
     ["speaker", "spatial-audio", "wifi", "smart-home"],
     "Six drivers · Dolby Atmos · Wi-Fi 6 · Trueplay tuning",
     "A single speaker that genuinely produces height and width, angling drivers sideways and upward. Pairs into a surround set with a Sonos soundbar."),

    # ---- phones ------------------------------------------------------------
    ("Apple iPhone 15 Pro 256GB", "Apple", "phones", "flagship", 1099, 4.8,
     ["smartphone", "titanium", "usb-c", "apple"],
     "6.1in 120Hz OLED · A17 Pro · 48MP main · titanium · USB-C",
     "Titanium frame drops meaningful weight from the previous generation, and the customisable Action button replaces the mute switch. The 48MP sensor finally shoots properly usable 24MP files by default."),
    ("Samsung Galaxy S24 Ultra 512GB", "Samsung", "phones", "flagship", 1419, 4.7,
     ["smartphone", "stylus", "telephoto", "android"],
     "6.8in 120Hz · Snapdragon 8 Gen 3 · 200MP · S Pen · 5x optical",
     "The most capable Android camera system available, with a 5x periscope that stays sharp in daylight and an S Pen tucked into the body. Big, heavy, and unapologetic about it."),
    ("Google Pixel 8a 128GB", "Google", "phones", "mid", 499, 4.6,
     ["smartphone", "camera", "android", "value"],
     "6.1in 120Hz · Tensor G3 · 64MP main · 7 years of updates",
     "Google's computational photography in a mid-range body, with seven years of OS and security updates — longer support than phones costing twice as much."),
    ("Nothing Phone (2a) 256GB", "Nothing", "phones", "entry", 349, 4.3,
     ["smartphone", "android", "budget", "design"],
     "6.7in 120Hz AMOLED · Dimensity 7200 Pro · 50MP dual · 45W",
     "Clean Android with a genuinely distinctive back panel. Fast where it matters — display, charging, day-to-day responsiveness — and honest about where it saves money."),

    # ---- laptops -----------------------------------------------------------
    ("Apple MacBook Air 15in M3 16GB/512GB", "Apple", "laptops", "flagship", 1499, 4.8,
     ["laptop", "macos", "portable", "apple-silicon"],
     "15.3in Liquid Retina · M3 · 18h battery · 1.51kg · fanless",
     "A large screen in a fanless, silent chassis that still runs all day on battery. The machine to buy if you want a big display without carrying a heavy laptop."),
    ("Dell XPS 14 (2024) Core Ultra 7", "Dell", "laptops", "flagship", 1699, 4.4,
     ["laptop", "windows", "oled", "creator"],
     "14.5in OLED 120Hz · Core Ultra 7 · RTX 4050 · 16GB · 1.7kg",
     "An OLED panel with real contrast and a discrete GPU in a chassis thin enough to commute with. The invisible haptic trackpad and capacitive function row divide opinion sharply."),
    ("Lenovo ThinkPad X1 Carbon Gen 12", "Lenovo", "laptops", "flagship", 1899, 4.6,
     ["laptop", "windows", "business", "keyboard"],
     "14in 2.8K OLED · Core Ultra 7 · 32GB · 1.09kg · MIL-STD",
     "Still the best keyboard on any laptop, in a carbon-fibre chassis that weighs almost nothing and survives being treated badly. Serviceable, with excellent Linux support."),
    ("ASUS Zenbook 14 OLED", "ASUS", "laptops", "mid", 899, 4.5,
     ["laptop", "windows", "oled", "value"],
     "14in 2.8K OLED 120Hz · Core Ultra 5 · 16GB · 1.2kg",
     "An OLED display and a full metal body at a price where most competitors still ship dim LCDs and plastic. The obvious choice for writing and browsing rather than heavy compute."),
    ("Framework Laptop 13 DIY Edition", "Framework", "laptops", "mid", 1049, 4.5,
     ["laptop", "repairable", "modular", "linux"],
     "13.5in 3:2 · Ryzen 7040 · swappable ports · user-serviceable",
     "Every part is replaceable with a screwdriver and a QR code, including the ports. Buy it if you intend to keep and upgrade a laptop for a decade rather than replace it."),

    # ---- cameras -----------------------------------------------------------
    ("Sony Alpha A7 IV Mirrorless Body", "Sony", "cameras", "flagship", 2499, 4.8,
     ["mirrorless", "full-frame", "hybrid", "video"],
     "33MP full-frame · 4K60 · 10-bit 4:2:2 · IBIS · dual card",
     "The default full-frame hybrid: reliable subject-tracking autofocus, genuinely usable 4K, and a sensor that holds detail well into high ISO. Heavy once a fast lens is attached."),
    ("Fujifilm X-T5 Body", "Fujifilm", "cameras", "flagship", 1699, 4.7,
     ["mirrorless", "aps-c", "retro", "photography"],
     "40MP APS-C · 7-stop IBIS · dedicated dials · 557g",
     "Physical dials for shutter speed and ISO mean you set exposure without entering a menu. Fuji's film simulations produce files you can use straight out of camera."),
    ("Canon EOS R8 Body", "Canon", "cameras", "mid", 1499, 4.5,
     ["mirrorless", "full-frame", "lightweight", "beginner"],
     "24MP full-frame · 4K60 oversampled · Dual Pixel AF II · 461g",
     "The lightest way into full-frame. Autofocus inherited from cameras costing three times as much, with compromises in battery life and a single card slot."),
    ("Sony FE 24-70mm f/2.8 GM II Lens", "Sony", "cameras", "flagship", 2299, 4.9,
     ["lens", "zoom", "full-frame", "professional"],
     "24-70mm · f/2.8 constant · 695g · weather-sealed",
     "The one lens that covers most professional work, rebuilt to be lighter and sharper than the original. Expensive, and the piece of glass most owners keep through several camera bodies."),
    ("DJI Osmo Pocket 3 Creator Combo", "DJI", "cameras", "mid", 799, 4.6,
     ["gimbal", "vlogging", "compact", "video"],
     "1in sensor · 3-axis gimbal · 4K120 · rotating touchscreen",
     "A stabilised 1-inch camera small enough to live in a pocket. Face tracking keeps you centred while walking, which is why it has largely replaced action cameras for talking-to-camera video."),

    # ---- tv ----------------------------------------------------------------
    ("LG C4 65in OLED evo TV", "LG", "tv", "flagship", 1799, 4.8,
     ["oled", "4k", "gaming", "hdr"],
     "65in OLED · 144Hz · 4x HDMI 2.1 · Dolby Vision · webOS",
     "Perfect blacks and per-pixel contrast, with four full-bandwidth HDMI 2.1 ports so a console and a PC can both run at high refresh. The reference choice for a dark room."),
    ("Samsung QN90D 55in Neo QLED TV", "Samsung", "tv", "flagship", 1299, 4.6,
     ["qled", "4k", "bright-room", "gaming"],
     "55in mini-LED · 144Hz · anti-glare · HDR10+ · Tizen",
     "Mini-LED backlighting gets far brighter than OLED, which matters in a room with windows. The matte anti-reflection layer is the reason to pick this over a cheaper panel."),
    ("Hisense U6N 55in Mini-LED TV", "Hisense", "tv", "entry", 549, 4.3,
     ["mini-led", "4k", "budget", "hdr"],
     "55in mini-LED · 60Hz · Dolby Vision · Google TV",
     "Genuine mini-LED backlighting and Dolby Vision at a price that used to buy a basic edge-lit panel. 60Hz limits it for gaming, but for film and television it punches far above its cost."),
    ("Sonos Arc Ultra Soundbar", "Sonos", "tv", "flagship", 999, 4.6,
     ["soundbar", "dolby-atmos", "wifi", "home-cinema"],
     "14 drivers · Dolby Atmos · eARC · Trueplay · Wi-Fi 6",
     "Height channels that actually reach the ceiling and come back, plus speech enhancement that rescues muttered dialogue. Expands with a sub and rear speakers later."),

    # ---- smart home --------------------------------------------------------
    ("Philips Hue White & Colour Starter Kit", "Philips", "smart-home", "mid", 179, 4.6,
     ["lighting", "zigbee", "smart-home", "matter"],
     "3 bulbs + bridge · 16M colours · Matter · Zigbee",
     "The lighting system everything else integrates with. The bridge keeps working when your internet does not, which is the difference between smart lighting and frustrating lighting."),
    ("Aqara Smart Hub M3 with Sensors", "Aqara", "smart-home", "mid", 129, 4.4,
     ["hub", "sensors", "matter", "automation"],
     "Matter controller · Thread border router · IR blaster · local",
     "Runs automations locally rather than in someone's cloud, bridges Zigbee and Thread devices into Matter, and replaces every infrared remote in the room."),
    ("Ecobee Smart Thermostat Premium", "Ecobee", "smart-home", "flagship", 249, 4.5,
     ["thermostat", "energy", "smart-home", "sensors"],
     "Room sensors · air quality monitor · Matter · built-in speaker",
     "Remote sensors mean the house heats to the temperature of the room you are in rather than the hallway. Pays for itself over a couple of winters in a badly balanced home."),
    ("Ring Battery Doorbell Plus", "Ring", "smart-home", "entry", 149, 4.2,
     ["doorbell", "camera", "security", "battery"],
     "1536p head-to-toe view · battery · two-way talk · no wiring",
     "Installs without touching mains wiring, and the taller sensor shows a whole person and any parcel on the doorstep rather than a cropped face."),

    # ---- gaming ------------------------------------------------------------
    ("Sony PlayStation 5 Slim Disc Edition", "Sony", "gaming", "flagship", 499, 4.7,
     ["console", "4k", "gaming", "disc"],
     "4K120 · ray tracing · 1TB SSD · DualSense haptics",
     "The haptic triggers change how games feel in a way screenshots cannot convey. The disc drive is detachable, so a physical library stays usable."),
    ("Valve Steam Deck OLED 1TB", "Valve", "gaming", "flagship", 649, 4.8,
     ["handheld", "pc-gaming", "linux", "portable"],
     "7.4in HDR OLED 90Hz · 1TB · 50Wh · SteamOS",
     "A full PC that plays your existing Steam library on a train. The OLED revision fixed the battery life and screen complaints of the original."),
    ("NVIDIA GeForce RTX 4070 Super 12GB", "NVIDIA", "gaming", "flagship", 599, 4.6,
     ["gpu", "pc-gaming", "ray-tracing", "dlss"],
     "12GB GDDR6X · DLSS 3 · 220W · 1440p high refresh",
     "The sensible high-refresh 1440p card. DLSS frame generation makes ray tracing playable at settings the raw silicon could not otherwise sustain."),
    ("Logitech G Pro X Superlight 2", "Logitech", "gaming", "mid", 159, 4.7,
     ["mouse", "wireless", "esports", "lightweight"],
     "60g · 32K DPI · 95h battery · USB-C · hybrid switches",
     "Sixty grams and no perceptible wireless latency. The mouse most competitive players actually use, which is unusual for a product marketed at them."),

    # ---- wearables ---------------------------------------------------------
    ("Apple Watch Series 10 46mm GPS", "Apple", "wearables", "flagship", 429, 4.7,
     ["smartwatch", "fitness", "health", "apple"],
     "Wide-angle OLED · ECG · sleep apnoea alerts · 18h · IP6X",
     "The thinnest Apple Watch with the largest display, adding sleep apnoea detection. Genuinely useful health monitoring, provided you carry an iPhone."),
    ("Garmin Forerunner 265 Music", "Garmin", "wearables", "mid", 449, 4.7,
     ["running", "gps", "amoled", "training"],
     "AMOLED · 13 days smartwatch / 20h GPS · offline music · multi-band",
     "Training load and recovery metrics that actually inform how you plan a week, with multi-band GPS that holds a track between tall buildings. Battery measured in weeks, not hours."),
    ("Oura Ring Gen 4 Silver", "Oura", "wearables", "mid", 349, 4.3,
     ["sleep", "recovery", "health", "ring"],
     "Up to 8 days · sleep staging · temperature trends · titanium",
     "Sleep and recovery tracking from something you forget you are wearing, which is the entire argument against a wrist device for sleep. Requires a subscription for the full analysis."),
]

DEMO_USERS = [
    ("admin@smartreco.dev", "admin12345", "Admin", "admin"),
    ("shopper@smartreco.dev", "shopper12345", "Alex Rivera", "user"),
    ("demo@smartreco.dev", "demo12345", "Demo User", "user"),
]


def seed(reset: bool = False) -> dict:
    configure_logging()
    init_db()

    if reset:
        from app.db import engine
        from app.services.vectorstore import get_vector_store

        log.warning("resetting all data")
        get_vector_store().reset()
        Base.metadata.drop_all(bind=engine)
        init_db()

    created_users, created_products = 0, 0

    with session_scope() as db:
        for email, password, name, role in DEMO_USERS:
            if db.scalar(select(User.id).where(User.email == email)):
                continue
            user = User(email=email, password_hash=hash_password(password), name=name, role=role)
            db.add(user)
            db.flush()
            db.add(UserProfile(user_id=user.id))
            created_users += 1

    with session_scope() as db:
        existing = {t for (t,) in db.execute(select(Product.title)).all()}
        for title, brand, category, tier, price, rating, tags, spec, description in PRODUCTS:
            if title in existing:
                continue
            create_product(
                db,
                ProductIn(
                    title=title,
                    description=description,
                    category=category,
                    tier=tier,
                    tags=tags,
                    price_cents=price * 100,
                    brand=brand,
                    spec=spec,
                    rating=rating,
                    is_published=True,
                ),
            )
            created_products += 1

    # One drain embeds every new product in a single batched call rather than one per row.
    synced = outbox.drain_all()
    health = outbox.health()

    log.info(
        "seed complete",
        extra={"users": created_users, "products": created_products, "synced": synced},
    )
    return {
        "users_created": created_users,
        "products_created": created_products,
        "sync": synced,
        "health": health,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the SmartReco demo catalogue")
    parser.add_argument("--reset", action="store_true", help="drop all data first")
    args = parser.parse_args()

    result = seed(reset=args.reset)
    print(f"\nusers created:    {result['users_created']}")
    print(f"products created: {result['products_created']}")
    print(f"vector sync:      {result['sync']}")
    print(f"in sync:          {result['health']['in_sync']} "
          f"(sql={result['health']['sql_published']} vectors={result['health']['vector_count']})")
    print("\nsign in as:")
    for email, password, _, role in DEMO_USERS:
        print(f"  {role:<5}  {email}  /  {password}")
