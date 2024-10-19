import os
import argparse
import random
from uuid import uuid4
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any, List, Tuple


def unique_random_string_decorator(func: Any):
    unique_values = {}

    def wrapper(*args: Any, **kwargs: Any) -> str:
        if func not in unique_values:
            unique_values[func] = set()

        while True:
            result = func(*args, **kwargs)
            if result not in unique_values[func]:
                unique_values[func].add(result)
                return result

    return wrapper


def generate_fullname() -> str:
    first_names = [
        "John",
        "Jane",
        "Michael",
        "Emily",
        "David",
        "Olivia",
        "Robert",
        "Sophia",
        "William",
        "Isabella",
        "James",
        "Ava",
        "Joseph",
        "Mia",
        "Charles",
        "Charlotte",
        "Thomas",
        "Amelia",
        "Daniel",
        "Harper",
    ]

    # Sample last names
    last_names = [
        "Smith",
        "Johnson",
        "Williams",
        "Brown",
        "Jones",
        "Davis",
        "Miller",
        "Wilson",
        "Moore",
        "Taylor",
        "Anderson",
        "Thomas",
        "Jackson",
        "White",
        "Harris",
        "Martin",
        "Thompson",
        "Garcia",
        "Martinez",
        "Lopez",
    ]

    random_first_name = random.choice(first_names)
    random_last_name = random.choice(last_names)
    return f"{random_first_name} {random_last_name}"


@unique_random_string_decorator
def generate_fullname_no_duplicate() -> str:
    return generate_fullname()


def generate_string_values(choices: List[str], min_count: int, max_count: int):
    if min_count > len(choices):
        raise ValueError(
            f"Only {len(choices)} choices available but {min} requested"
        )
    count = random.randint(min_count, min(max_count, len(choices)))
    result = random.sample(choices, count)
    return ",".join(result)


@dataclass
class FrameObject:
    object_id: str
    video_id: str
    video_name: str
    type: str
    category: str
    tracking_id: str
    timestamp: int


tracking_ids: List[Tuple[str, str]] = []


@dataclass
class VideoFrame:
    frame_id: str
    video_id: str
    video_name: str
    timestamp: int
    objects: List[FrameObject]


@dataclass
class VideoShot:
    shot_id: str
    video_id: str
    video_name: str
    description: str
    start: int
    finish: int


@dataclass
class VideoTranscript:
    transcript_id: str
    video_id: str
    video_name: str
    text: str
    start: int
    finish: int


@dataclass
class Video:
    video_id: str
    video_name: str
    type: str
    summary: str
    duration: int
    frames: List[VideoFrame]
    shots: List[VideoShot]
    transcripts: List[VideoTranscript]

    videos = [
        (
            "Exploring the Amazon Rainforest",
            "Join us on a thrilling adventure through the heart of the Amazon rainforest, home to incredible biodiversity.",
        ),
        (
            "The Quest for Mars",
            "A deep dive into space exploration and the journey to send humans to the Red Planet.",
        ),
        (
            "Saving Endangered Species",
            "Discover the efforts to protect and preserve the world's most endangered animals.",
        ),
        (
            "The Art of Filmmaking",
            "An insider's look at the magic behind the scenes of your favorite movies.",
        ),
        (
            "Epic World History",
            "Travel through time and explore significant historical events and figures.",
        ),
        (
            "Culinary Delights of Italy",
            "Savor the flavors of Italy with mouthwatering pasta, pizza, and gelato.",
        ),
        (
            "Wildlife Under the Sea",
            "Dive into the mysterious depths of the ocean and meet its fascinating inhabitants.",
        ),
        (
            "Inspirational Stories of Resilience",
            "Real-life tales of individuals who overcame adversity and found success.",
        ),
        (
            "The Universe Unveiled",
            "A cosmic journey exploring galaxies, black holes, and the mysteries of the universe.",
        ),
        (
            "Heartfelt Family Reunions",
            "Heartwarming stories of long-lost family members reuniting against all odds.",
        ),
        (
            "The Enchanted Forest",
            "An animated adventure filled with magical creatures and whimsical wonders.",
        ),
        (
            "Spectacular Sports Moments",
            "Relive the most thrilling and unforgettable moments in sports history.",
        ),
        (
            "Historical Love Stories",
            "Romantic tales from different eras that will tug at your heartstrings.",
        ),
        (
            "Crime Solvers: Unsolved Mysteries",
            "Put on your detective hat and try to crack these perplexing cold cases.",
        ),
        (
            "Architectural Wonders",
            "Explore the world's most iconic and breathtaking architectural masterpieces.",
        ),
        (
            "A Journey Through Japan",
            "An immersive experience into Japan's rich culture, traditions, and landscapes.",
        ),
        (
            "Laugh Out Loud Comedy Show",
            "Get ready for a night of laughter with top comedians delivering hilarious stand-up.",
        ),
        (
            "Survival in the Wild",
            "Learn essential survival skills from experts in extreme outdoor environments.",
        ),
        (
            "Epic Fantasy Adventures",
            "Step into magical realms with mythical creatures, heroes, and epic quests.",
        ),
        (
            "Innovation and Technology Today",
            "Discover the latest advancements in technology and how they shape our future.",
        ),
        (
            "Mysteries of Ancient Egypt",
            "Unlock the secrets of the pyramids, pharaohs, and the Egyptian civilization.",
        ),
        (
            "Inside the Human Mind",
            "A fascinating exploration of psychology and the complexities of human behavior.",
        ),
        (
            "Beautiful World Music",
            "Experience the sounds of world music and diverse musical traditions.",
        ),
        (
            "Legal Battles in the Courtroom",
            "Intense courtroom dramas filled with riveting trials, legal strategies, and verdicts.",
        ),
        (
            "A Taste of Home",
            "Heartwarming stories of people returning to their roots and finding their place in the world.",
        ),
        (
            "High-Stakes Heist Operation",
            "Mastermind criminals execute daring heists with intricate plans and daring schemes.",
        ),
        (
            "The Secrets of Science",
            "A journey through scientific discoveries, experiments, and the wonders of the natural world.",
        ),
        (
            "Light-Hearted Romantic Comedy",
            "Romantic comedies that will make you smile, laugh, and believe in love.",
        ),
        (
            "Edge of Your Seat Thrillers",
            "Nail-biting suspense, unexpected plot twists, and thrilling mysteries.",
        ),
        (
            "Exploring World Art",
            "Delve into the world of art, from classical masterpieces to contemporary creations.",
        ),
        (
            "Green Earth: Sustainable Living",
            "Learn about eco-friendly practices and environmental conservation.",
        ),
        (
            "Ancient Myths and Legends",
            "Uncover the mythical tales, gods, and legendary creatures from ancient cultures.",
        ),
        (
            "The Quest for Adventure",
            "Adrenaline-packed adventures that take you to the farthest reaches of the globe.",
        ),
        (
            "Emotional Human Journeys",
            "Powerful stories of human experiences, relationships, and personal growth.",
        ),
        (
            "Mind-Bending Thrillers",
            "Psychological and suspenseful thrillers that challenge your perception of reality.",
        ),
        (
            "A World of Science",
            "Fascinating documentaries on groundbreaking scientific research and discoveries.",
        ),
        (
            "Hilarious Sitcom Classics",
            "Laugh along with iconic sitcoms that have kept audiences entertained for years.",
        ),
        (
            "The Music of Nature",
            "Relax and unwind with the soothing sounds of nature and ambient music.",
        ),
        (
            "Planet Earth's Beauty",
            "A visual feast capturing the awe-inspiring beauty of our planet's natural landscapes.",
        ),
        (
            "Epic Space Exploration",
            "Journey to the stars with tales of space travel, cosmic phenomena, and otherworldly wonders.",
        ),
    ]

    _transcripts = [
        "In this episode, we delve into the fascinating world of artificial intelligence. Our guest, Dr. Sarah Anderson, shares insights into the latest advancements in AI and its potential impact on various industries. We discuss the ethical considerations of AI, its applications in healthcare, and the future of AI research. Tune in to gain a deeper understanding of this transformative technology.",
        "Welcome to the podcast! Today, we're joined by renowned chef, Julia Martinez, who takes us on a culinary journey through the flavors of Spain. She shares her love for traditional Spanish dishes, from paella to tapas. Get ready to be inspired to cook your own Spanish delicacies at home as Julia provides tips and tricks to master these mouthwatering recipes.",
        "In this episode, we explore the world of mindfulness and meditation with mindfulness coach, David Johnson. Discover the many benefits of incorporating mindfulness practices into your daily life. From reducing stress and anxiety to enhancing focus and productivity, David shares practical tips for beginners and advanced practitioners alike.",
        "Join us as we travel back in time to explore ancient civilizations. Dr. Emily Roberts, an archaeologist, guides us through the mysteries of ancient Egypt. Learn about the construction of the pyramids, deciphering hieroglyphics, and the daily life of the pharaohs. Get ready to embark on a historical adventure like no other.",
        "Ever wondered about the power of storytelling? In this episode, we sit down with award-winning author, James Thompson, who shares his experiences and insights into the world of literature. We discuss the art of crafting compelling narratives, his writing process, and the impact of storytelling on society. Whether you're an aspiring author or a book lover, this episode is a must-listen.",
        "Our guest today is Dr. Laura Adams, a clinical psychologist, who sheds light on the importance of mental health. We explore the stigma surrounding mental health issues, strategies for managing stress and anxiety, and the role of therapy in personal growth. Join us for an insightful conversation that promotes mental well-being.",
        "Welcome to a journey through the cosmos! Dr. John Parker, an astrophysicist, takes us on a thrilling adventure to explore the mysteries of the universe. We discuss black holes, the search for extraterrestrial life, and the latest breakthroughs in space exploration. Prepare to be amazed by the wonders of the cosmos.",
        "In this episode, we explore the world of sustainable living with eco-entrepreneur, Lisa Davis. Discover eco-friendly practices, from reducing waste and conserving energy to embracing green technologies. We discuss the impact of climate change and the role of individuals in creating a more sustainable future.",
        "Join us for a conversation with fitness expert, Mark Johnson, as we discuss the keys to a healthy lifestyle. Mark shares his expertise on exercise, nutrition, and maintaining a balanced life. Get ready to be inspired to take charge of your health and well-being.",
        "Have you ever dreamed of traveling the world? Travel blogger, Sarah Collins, shares her experiences and tips for globetrotters. From hidden gems to travel hacks, we cover it all. Whether you're planning your next adventure or simply love hearing travel stories, this episode will ignite your wanderlust.",
        "In this episode, we dive into the world of technology and innovation. Tech guru, Chris Wilson, discusses the latest gadgets, trends in the tech industry, and the impact of technology on our daily lives. Stay updated on the tech world with our insightful conversation.",
        "Welcome to a world of creativity! Join us for an inspiring chat with renowned artist, Emma Turner. Emma shares her artistic journey, from her early beginnings to her latest projects. Discover the power of artistic expression and the beauty of visual storytelling.",
        "Ever wondered about the world of entrepreneurship? In this episode, we sit down with successful business owner, Alex Turner. From launching a startup to overcoming challenges, we explore the entrepreneurial journey. Whether you're an aspiring entrepreneur or simply curious about the business world, this episode offers valuable insights.",
        "Join us as we explore the world of photography with acclaimed photographer, Olivia Walker. We discuss the art of capturing moments, techniques for taking stunning photos, and the impact of visual storytelling. Whether you're a professional photographer or an amateur enthusiast, you'll find inspiration in this episode.",
        "In this episode, we journey through history with historian, Dr. Robert Davis. We delve into fascinating historical events, figures, and periods that have shaped our world. From ancient civilizations to modern revolutions, history comes to life in this captivating conversation.",
        "Travel back in time with archaeologist, Sarah Turner, as we uncover the mysteries of ancient Rome. Learn about the architecture, culture, and daily life of this iconic civilization. If you're a history enthusiast or simply curious about the past, this episode is a treasure trove of knowledge.",
        "Welcome to a new episode of 'Exploring Nature.' Join us as we embark on an adventure through the Amazon Rainforest. Our guide, Maria Hernandez, introduces us to the diverse wildlife and rich biodiversity of this unique ecosystem. Learn about the conservation efforts to protect this vital part of our planet.",
        "In 'Tech Trends,' we discuss the latest innovations in the world of technology. Our host, John Roberts, explores topics such as virtual reality, 5G networks, and the impact of technology on everyday life. Get ready for an insightful journey into the rapidly changing tech landscape.",
        "Are you a book lover? If so, you won't want to miss this episode of 'Bookworm's Paradise.' Join our host, Emma Taylor, as she reviews some of the most captivating books of the year. From thrilling mysteries to heartwarming romances, we've got something for every reader.",
        "Discover the world of culinary delights in 'Taste of Culture.' In this episode, our chef, Carlos Rodriguez, takes you on a culinary journey to explore international cuisines. From spicy curries to delicate pastries, you'll experience a symphony of flavors from around the globe.",
    ]

    @staticmethod
    def get_existing_tracking_ids(type: str):
        result: List[str] = []
        for existing_type, tracking_id in tracking_ids:
            if existing_type == type:
                result.append(tracking_id)
        return result

    @staticmethod
    def get_tracking_id(type: str) -> str:
        should_track = random.randint(0, 10)
        if should_track > 0:
            return ""

        existing = Video.get_existing_tracking_ids(type)
        if existing:
            return random.choice(existing)

        if len(tracking_ids) >= 10:
            tracking_ids.pop(random.randint(0, len(tracking_ids) - 1))

        tracking_id = str(uuid4())
        tracking_ids.append((type, tracking_id))
        return tracking_id

    @staticmethod
    def generate_frame(
        video_id: str, video_name: str, timestamp: int
    ) -> VideoFrame:
        frame_id = f"{video_id}_{timestamp}"
        objects: List[FrameObject] = []
        object_count = random.randint(0, 3)
        for i in range(object_count):
            object_id = f"{frame_id}_{i}"
            type, category = random.choice(
                [
                    ("person", ""),
                    ("dog", "pet"),
                    ("cat", "pet"),
                    ("car", "vehicle"),
                    ("monitor", "electronics"),
                ]
            )
            tracking_id = Video.get_tracking_id(type)
            objects.append(
                FrameObject(
                    object_id,
                    video_id,
                    video_name,
                    type,
                    category,
                    tracking_id,
                    timestamp,
                )
            )
        return VideoFrame(frame_id, video_id, video_name, timestamp, objects)

    @staticmethod
    def generate(index: int):
        video_id = f"video{index}"
        video_name, summary = Video.videos[index]
        duration = random.randint(60, 60 * 10)

        frames: List[VideoFrame] = []
        for i in range(duration):
            timestamp = i * 1000
            frames.append(Video.generate_frame(video_id, video_name, timestamp))

        shots: List[VideoShot] = []
        shot_duration = 0
        while shot_duration + 30 < duration:
            start = random.randint(shot_duration, shot_duration + 20)
            finish = random.randint(start + 1, duration - 1)
            description = f"Shot from {start}s to {finish}s of {video_name}"
            shots.append(
                VideoShot(
                    f"shot_{video_id}_{start}_{finish}",
                    video_id,
                    video_name,
                    description,
                    start * 1000,
                    finish * 1000,
                )
            )
            shot_duration = finish

        transcripts: List[VideoTranscript] = []
        transcript_duration = 0
        while transcript_duration + 30 < duration:
            start = random.randint(
                transcript_duration, transcript_duration + 20
            )
            finish = random.randint(start + 1, duration - 1)
            text = random.choice(Video._transcripts)
            transcripts.append(
                VideoTranscript(
                    f"transcript_{video_id}_{start}_{finish}",
                    video_id,
                    video_name,
                    text,
                    start * 1000,
                    finish * 1000,
                )
            )
            transcript_duration = finish

        return Video(
            video_id,
            video_name,
            "mp4",
            summary,
            duration * 1000,
            frames,
            shots,
            transcripts,
        )


@dataclass
class User:
    user_id: str
    full_name: str
    is_creator: bool
    is_influencer: bool
    country_name: str
    associated_hashtags: str
    interests: str
    birth_year: int
    birth_month: int
    birth_day: int

    @staticmethod
    def generate(index: int):
        user_id = f"user{index}"
        full_name: str = generate_fullname_no_duplicate()
        is_creator = random.choice([True, False])
        is_influencer = random.choice([True, False])
        country_name = random.choice(
            ["USA", "UK", "France", "Singapore", "India", "China"]
        )
        associated_hashtags = generate_string_values(
            ["technical", "leisure", "fashion"], 1, 2
        )
        interests = generate_string_values(
            ["AI", "travel", "cooking", "chess"], 1, 3
        )
        birthday = datetime.now() - timedelta(
            days=random.randint(15 * 365, 30 * 365)
        )
        return User(
            user_id,
            full_name,
            is_creator,
            is_influencer,
            country_name,
            associated_hashtags,
            interests,
            birthday.year,
            birthday.month,
            birthday.day,
        )


@dataclass
class Episode:
    episode_id: str
    podcast_id: str
    podcast_name: str
    episode_title: str
    episode_description: str
    video_name: str
    creation_time: str


@dataclass
class Podcast:
    podcast_id: str
    podcast_name: str
    description: str
    host_name: str
    episodes: List[Episode]

    _podcasts = [
        (
            "Tech Talk Daily",
            "Stay updated with the latest tech trends and innovations from around the world.",
        ),
        (
            "Business Buzz",
            "Dive deep into the world of entrepreneurship and discover success stories and strategies.",
        ),
        (
            "True Crime Chronicles",
            "Explore gripping true crime stories and delve into the minds of notorious criminals.",
        ),
        (
            "Mindful Living",
            "Learn mindfulness techniques and practices to lead a happier and more balanced life.",
        ),
        (
            "Planet Earth Unplugged",
            "Discover the wonders of our planet's natural beauty and the incredible creatures that inhabit it.",
        ),
        (
            "Foodie Adventures",
            "Embark on a culinary journey exploring diverse cuisines and tantalizing recipes.",
        ),
        (
            "The Comedy Corner",
            "Get ready for a good laugh with stand-up comedians and hilarious sketches.",
        ),
        (
            "Historical Mysteries",
            "Unearth unsolved mysteries from history and uncover the secrets of the past.",
        ),
        (
            "Fictional Worlds",
            "Immerse yourself in epic tales and imaginary realms created by talented authors.",
        ),
        (
            "Career Insights",
            "Gain valuable career advice, interview tips, and insights to advance your professional life.",
        ),
        (
            "Wellness Wisdom",
            "Prioritize your well-being with discussions on mental health, fitness, and holistic healing.",
        ),
        (
            "The Space Explorer",
            "Journey through the cosmos, exploring the mysteries of space and celestial bodies.",
        ),
        (
            "Political Pulse",
            "Stay informed about current events and political developments shaping the world.",
        ),
        (
            "Traveler's Tales",
            "Travel vicariously through captivating stories and experiences from globetrotters.",
        ),
        (
            "Science Simplified",
            "Learn about complex scientific concepts made easy to understand for everyone.",
        ),
        (
            "Family Matters",
            "Navigate the challenges and joys of family life with insightful advice and stories.",
        ),
        (
            "Hidden Gems",
            "Discover lesser-known movies, books, and music that deserve more recognition.",
        ),
        (
            "The Art Gallery",
            "Explore the world of art, including famous artworks, artists, and creative movements.",
        ),
        (
            "Sustainable Living",
            "Find inspiration for eco-friendly living and sustainable practices for a greener future.",
        ),
        (
            "Sports Spectacle",
            "Dive into the world of sports, featuring thrilling games, athletes, and sports history.",
        ),
    ]

    _episodes = [
        (
            "The Journey Begins",
            "Embark on a thrilling journey into the unknown as our heroes set out on a quest of a lifetime.",
        ),
        (
            "Uncharted Territory",
            "Explore mysterious lands that few have ventured into, filled with unexpected challenges and treasures.",
        ),
        (
            "Lost and Found",
            "When hope seems lost, unexpected discoveries bring hope back to the hearts of the adventurers.",
        ),
        (
            "Secrets of the Past",
            "Unearth long-buried secrets that will change the course of history and unravel the past's mysteries.",
        ),
        (
            "Into the Abyss",
            "Venture into the depths of the abyss where danger and intrigue await at every turn.",
        ),
        (
            "Whispers in the Dark",
            "Whispers in the dark signal an impending threat, and the heroes must decipher their cryptic messages.",
        ),
        (
            "The Forgotten Realm",
            "Discover a realm that time has forgotten, teeming with ancient wonders and forgotten civilizations.",
        ),
        (
            "Echoes of Eternity",
            "Echoes of eternity resonate through time, guiding our heroes toward their destiny.",
        ),
        (
            "A New Dawn",
            "With the dawn of a new day, new possibilities and challenges arise on the horizon.",
        ),
        (
            "Shadows of Reality",
            "Reality becomes a shadowy place as the heroes face the mysteries that lie just beyond the surface.",
        ),
        (
            "The Enchanted Forest",
            "Step into the enchanting forest, where magical creatures and mythical beings await your arrival.",
        ),
        (
            "The Road Less Traveled",
            "Choosing the road less traveled leads to unexpected encounters and extraordinary tales.",
        ),
        (
            "Legends and Myths",
            "Myths and legends come to life as our heroes embark on legendary quests and meet legendary figures.",
        ),
        (
            "Mysteries of the Deep",
            "Journey into the depths of the deep, where strange creatures and forgotten relics await.",
        ),
        (
            "Realm of Dreams",
            "The realm of dreams is a place of wonder, but it's not without its own set of dangers.",
        ),
        (
            "Forging Alliances",
            "Forging alliances is key to overcoming challenges and achieving great feats in the land of adventure.",
        ),
        (
            "The Quest for Truth",
            "The quest for truth is never-ending as heroes seek to unravel the mysteries of their world.",
        ),
        (
            "Rise of the Phoenix",
            "The phoenix rises from the ashes, signifying a time of renewal and transformation.",
        ),
        (
            "Tales from Beyond",
            "Tales from beyond bring stories of distant lands, each with its own unique charm.",
        ),
        (
            "The Hidden Key",
            "The hidden key unlocks secrets that were meant to remain concealed.",
        ),
        (
            "Through the Veil",
            "Venture through the veil and experience a world beyond imagination.",
        ),
        (
            "Songs of Destiny",
            "Songs of destiny guide the heroes on their journey, helping them uncover the truth.",
        ),
        (
            "Echoes from the Past",
            "Echoes from the past resonate with the present, offering valuable lessons for the future.",
        ),
        (
            "The Dark Prophecy",
            "A dark prophecy looms over the land, and heroes must confront the shadows it casts.",
        ),
        (
            "Cursed Relics",
            "Cursed relics hold both power and danger, and heroes must make crucial choices.",
        ),
        (
            "Voyage of Discovery",
            "A voyage of discovery takes our heroes to new lands, filled with wonders and enigmas.",
        ),
        (
            "Riddles and Enigmas",
            "Riddles and enigmas present themselves at every turn, challenging the minds of our adventurers.",
        ),
        (
            "Chronicles of Valor",
            "The chronicles of valor tell the stories of bravery, courage, and heroism.",
        ),
        (
            "Whispers of the Ancients",
            "Whispers of the ancients offer cryptic guidance as heroes uncover the secrets of the past.",
        ),
        (
            "Sands of Time",
            "The sands of time flow inexorably, reminding our heroes of the fleeting nature of life.",
        ),
        (
            "Eternal Echoes",
            "Eternal echoes resonate throughout the ages, connecting the past, present, and future.",
        ),
        (
            "The Alchemist's Secret",
            "The alchemist's secret holds the key to unlocking great power and potential.",
        ),
        (
            "The Lost Scroll",
            "The lost scroll reveals ancient knowledge that could change the course of history.",
        ),
        (
            "Guardians of the Kingdom",
            "Guardians of the kingdom watch over and protect the land from threats and invaders.",
        ),
        (
            "The Forbidden Land",
            "The forbidden land is a place of mystery and danger, where few dare to tread.",
        ),
        (
            "The Crystal Guardians",
            "The crystal guardians are powerful beings tasked with safeguarding precious treasures.",
        ),
        (
            "The Enigma Code",
            "The enigma code conceals secrets that only the most determined can decipher.",
        ),
        (
            "Pirates of the Abyss",
            "Pirates of the abyss sail the treacherous seas, seeking adventure and fortune.",
        ),
        (
            "The Cryptic Chronicles",
            "Explore the cryptic chronicles of a world filled with enigmas and surprises.",
        ),
        (
            "The Final Frontier",
            "The heroes embark on their ultimate journey, facing challenges in the final frontier.",
        ),
    ]

    @staticmethod
    def generate(
        podcast_index: int,
        episode_index: int,
        episode_count: int,
        videos: List[Video],
    ):
        podcast_id = f"podcast{podcast_index}"
        if podcast_index < len(Podcast._podcasts):
            podcast_name, description = Podcast._podcasts[podcast_index]
        else:
            podcast_name = f"podcast #{podcast_index}"
            description = "This is a podcast of unknown topic"
        host_name = generate_fullname()

        episodes: List[Episode] = []
        for _ in range(episode_count):
            episode_title, episode_description = Podcast._episodes[
                episode_index
            ]
            episode_title = podcast_name + ": " + episode_title
            episode_description = podcast_name + ": " + episode_description
            video_id = videos[episode_index].video_id
            video_name = videos[episode_index].video_name
            creation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            episodes.append(
                Episode(
                    f"episode_{video_id}",
                    podcast_id,
                    podcast_name,
                    episode_title,
                    episode_description,
                    video_name,
                    creation_time,
                )
            )
            episode_index += 1

        return Podcast(
            podcast_id, podcast_name, description, host_name, episodes
        )


@dataclass
class EpisodeEngagement:
    episode_id: str
    podcast_id: str
    podcast_name: str
    user_id: str
    video_name: str
    reaction: str
    comment: str
    shares: int
    impressions: int
    engagement_timestamp: int

    _comments = [
        "This video explained the concept so clearly. I finally get it!",
        "The visuals in this video were amazing. I was glued to the screen.",
        "I learned something new today. Thanks for the informative video!",
        "I can't believe I've been missing out on this content. Subscribed!",
        "The presenter's sense of humor made this video enjoyable and fun.",
        "I watched this video twice, it's that good! Kudos to the creator.",
        "The practical tips in this video will definitely come in handy.",
        "I shared this video with my friends, it's too good to keep to myself.",
        "I appreciate the effort put into making this video. Well done!",
        "The comments section brought me here, and I'm not disappointed.",
    ]

    @staticmethod
    def generate(episode: Episode, video: Video, user: User):
        duration = video.duration // 1000
        timestamp = random.randint(0, duration - 1) * 1000
        reaction = random.choice(["positive", "negative", "neutral"])
        comment = random.choice(EpisodeEngagement._comments)
        shares = random.randint(0, 10)
        impressions = random.randint(0, 10)
        return EpisodeEngagement(
            episode.episode_id,
            episode.podcast_id,
            episode.podcast_name,
            user.user_id,
            episode.video_name,
            reaction,
            comment,
            shares,
            impressions,
            timestamp,
        )

os.chdir(r"C:\Users\anyas\Desktop\Summer Project\SQL CSVs")

def output_csv(data: List[Any], filename: str):
    with open(filename, "w") as file:
        for i, item in enumerate(data):
            line = ""
            header = ""

            for key, value in vars(item).items():
                if isinstance(key, str) and not isinstance(value, list):
                    if i == 0:
                        header += key + ","

                    quota = '"' if isinstance(value, str) else ""
                    line += f"{quota}{value}{quota},"
            if i == 0:
                file.write(header.rstrip(",") + "\n")
            file.write(line.rstrip(",") + "\n")


def generate_users(user_count: int, dataset_name: str):
    users: List[User] = []
    for i in range(user_count):
        users.append(User.generate(i))

    output_csv(users, os.path.join(dataset_name, "user.csv"))
    return users


def generate_podcasts(
    podcast_count: int, videos: List[Video], dataset_name: str
):
    podcasts: List[Podcast] = []
    episodes: List[Episode] = []
    for i in range(podcast_count):
        episodes_left = len(Video.videos) - len(episodes)
        podcasts_left = podcast_count - i
        if podcasts_left == 1:
            episode_count = episodes_left
        else:
            avg_episode = episodes_left // podcasts_left
            max_episode = min(
                avg_episode * 2, episodes_left - (podcasts_left - 1)
            )
            episode_count = random.randint(1, max_episode)
        print(f"episode count={episode_count} episodes_left={episodes_left}")

        podcasts.append(
            Podcast.generate(i, len(episodes), episode_count, videos)
        )
        episodes.extend(podcasts[i].episodes)

    output_csv(podcasts, os.path.join(dataset_name, "podcast.csv"))
    output_csv(episodes, os.path.join(dataset_name, "episode.csv"))
    return (podcasts, episodes)


def generate_videos(prefix: str, dataset_name: str):
    videos: List[Video] = []
    frames: List[VideoFrame] = []
    objects: List[FrameObject] = []
    shots: List[VideoShot] = []
    transcripts: List[VideoTranscript] = []
    for i in range(len(Video.videos)):
        videos.append(Video.generate(i))
        frames.extend(videos[i].frames)
        shots.extend(videos[i].shots)
        transcripts.extend(videos[i].transcripts)
        for frame in videos[i].frames:
            objects.extend(frame.objects)

    output_csv(videos, os.path.join(dataset_name, prefix + "videos.csv"))
    output_csv(frames, os.path.join(dataset_name, prefix + "videos_frames.csv"))
    output_csv(
        objects, os.path.join(dataset_name, prefix + "videos_objects.csv")
    )
    output_csv(shots, os.path.join(dataset_name, prefix + "videos_shots.csv"))
    output_csv(
        transcripts,
        os.path.join(dataset_name, prefix + "videos_transcripts.csv"),
    )
    return videos


def generate_engagements(
    engagement_count: int,
    episodes: List[Episode],
    videos: List[Video],
    users: List[User],
    dataset_name: str,
):
    engagements: List[EpisodeEngagement] = []
    for _ in range(engagement_count):
        episode_index = random.randint(0, len(episodes) - 1)
        episode = episodes[episode_index]
        if episode.video_name:
            engagements.append(
                EpisodeEngagement.generate(
                    episode, videos[episode_index], random.choice(users)
                )
            )

    output_csv(engagements, os.path.join(dataset_name, "episodeengagement.csv"))
    return engagements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--podcasts", type=int, default=10)
    parser.add_argument("--engagements_per_episode", type=int, default=10)
    parser.add_argument("--prefix", default="episode")
    parser.add_argument("--dataset_name", default="podcasts")
    args = parser.parse_args()

    random.seed(args.seed)

    videos = generate_videos(args.prefix, args.dataset_name)
    users = generate_users(args.users, args.dataset_name)
    _, episodes = generate_podcasts(args.podcasts, videos, args.dataset_name)
    generate_engagements(
        args.engagements_per_episode * len(episodes),
        episodes,
        videos,
        users,
        args.dataset_name,
    )


if __name__ == "__main__":
    main()
#%%
import os
import pandas as pd
import sqlite3

folder_path = r"C:\Users\anyas\Desktop\Summer Project\SQL CSVs\podcasts"

sqlite_db = "episodes.db"

conn = sqlite3.connect(sqlite_db)

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        
        table_name = os.path.splitext(filename)[0]
        
        df = pd.read_csv(file_path)
        
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        print(f"Table '{table_name}' created in SQLite database '{sqlite_db}'")

conn.close()

print("All CSV files have been imported into the SQLite database.")