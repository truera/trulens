import json

answer_relevance_golden_set = [
    {
        "query": "How many stomachs does a cow have?",
        "response": "Cows' diet relies primarily on grazing.",
        "expected_score": 0.2
    }, {
        "query":
            "Name some famous dental floss brands",
        "response":
            "Oral-B is an American brand of oral hygiene products, including toothpastes, toothbrushes, electric toothbrushes, and mouthwashes. The brand has been in business since the invention of the Hutson toothbrush in 1950 and in Redwood City, California.",
        "expected_score":
            0.4
    }, {
        "query":
            "Name some famous floss brands",
        "response":
            "Types of floss and alternative options. Dental floss is regarded as the gold standard — it’s been around the longest compared to other plaque-removing products, Hewlett said. Moursi also added that most flossing research studies have been conducted with dental floss, so there’s a lot of data showing its effectiveness. But floss is not one-size-fits-all, he noted. Since using dental floss is difficult for some, there are other effective tools like interdental cleaners. Below, we broke down the differences among several different options. Dental floss When people think of dental floss, it’s usually the threaded variety that comes on a spool. But there’s also dental tape, which Hewlett described as a wider and flatter type of floss. He said it's particularly useful for people with larger spaces between their teeth since it covers more surface area. Both forms of floss come in unflavored or flavored varieties, but choosing a flavored option has no impact on how well it cleans your teeth, Hewlett said. Flosses also come waxed and unwaxed — while a wax coating can make floss pass between teeth more easily, Hewitt said, both waxed and unwaxed are equally effective when used properly. Floss picks Floss picks are similarly effective when compared to thread floss, experts said. The picks look like a wand and have a small piece of floss at the forked end, so you can grip the handle while using the tool. Experts said floss picks are generally easy to use, especially if you’re flossing a child’s teeth. Water flossers Water flossers are powered devices that shoot pressurized water at the spaces between teeth, targeting debris to disrupt and flush out plaque. While there is evidence to support their ability to remove plaque from teeth, Moursi said for water flossers to do their job, “you have to hold it in just the right place, at just the right angle and for just the right amount of time,” which can be challenging. Anyone can use water flossers, but experts said they’re the most beneficial for people who have difficulty using thread floss or floss threaders, as well as those with certain dental work like braces, bridges and crowns. Interdental brushes Dental work like braces, bridges and crowns can block floss from slipping between teeth, making flossing challenging. Interdental brushes — which look like little spoolie brushes — can pass through the spaces between teeth and under any dental work, allowing you to remove plaque. The brushes have bristles on one end and a handle to grip on the other. To use, you point the brush at the gum line between teeth and push it through, moving the bristles around the space to remove plaque, said Hewlett. The brushes come in various shapes and sizes to fit the spaces between your teeth.",
        "expected_score":
            0.4
    }, {
        "query":
            "How does the social structure of a lion pride impact the genetic diversity and long-term survival of the species?",
        "response":
            "A typical pride of lions consists of about six related females, their dependent offspring, and a “coalition” of 2–3 resident males that joined the pride from elsewhere. The pride is a “fission-fusion” society and pridemates are seldom found together, except for mothers that have pooled their offspring into a “crèche\".",
        "expected_score":
            0.7
    }, {
        "query":
            "What are the parts of a cow's digestive tract, and how do they work including mouth, esophagus, the stomach, and intestines.",
        "response":
            "The cow's digestive tract consists of the following.\n\nMouth\nA four-compartment stomach, which includes\nThe rumen (paunch)\nThe reticulum (“honeycomb”)\nThe omasum (“manyplies”)\nThe abomasum (“true stomach”)\nSmall intestine\nLarge intestine\n\nThe rumen\nThe rumen (on the left side of the animal) is the largest stomach compartment and consists of several sacs. It can hold 25 gallons or more of material depending on the size of the cow. Because of its size, the rumen acts as a storage or holding vat for feed.\n\nAside from storage, the rumen is also a fermentation vat. The rumen’s environment favors the growth of microbes. These microbes digest or ferment feed within the rumen and make volatile fatty acids (VFAs). The rumen absorbs most of the VFAs from fermentation.\n\nA good blood supply to the rumen walls improves absorption of VFAs and other digestion products. Tiny projections (papillae) line the rumen, which increases the rumen’s surface area and the amount it can absorb.\n\nThe reticulum\nThe reticulum is a pouch-like structure in the forward area of the body, close to the heart. The tissues in the reticulum form a network similar to a honeycomb. A small tissue fold lies between the reticulum and rumen, but the two aren’t separate compartments. Together they’re called the rumino-reticulum.\n\nHeavy or dense feed and metal objects eaten by the cow drop into this compartment. Nails and other sharp objects may work into the tissue and cause “hardware disease.” You can use magnets to prevent disease or correct the problem through surgery. Leaving it untreated may lead to infection and possibly death.\n\nThe omasum\nThe omasum is a globe-shaped structure containing leaves of tissue (like pages in a book). It absorbs water and other substances from digestive contents. Feed material (ingesta) between the leaves will be drier than ingesta found in the other compartments.\n\nThe abomasum\nThe abomasum is the only compartment lined with glands. These glands release hydrochloric acid and digestive enzymes, needed to breakdown feeds. The abomasum is similar to a nonruminant stomach.\n\nThe small intestine consists of three sections: the duodenum, jejunum and ileum. It measures about 20 times the length of the animal.\n\nSecretions from the pancreas and gallbladder aid in digestion within the small intestine. The small intestine completes most of the digestive process and absorbs many nutrients through villi (small finger-like projections). From the villi the nutrients enter into the blood and lymphatic systems.\n\nThe cecum is the large area where the small and large intestine meet. The cecum breaks down some previously undigested fiber, but the exact importance of the cecum remains unknown.\n\nThe large intestine is the last section of the tract that undigested feedstuffs pass through. Microbes digest some undigested feed here, but the main digestive function of the large intestine is to absorb water.",
        "expected_score":
            0.8
    }, {
        "query":
            "Is Denny's open right now?",
        "response":
            "As an AI assistant, I don't have access to realtime information. You should consult the web for answers about restaurant opening times.",
        "expected_score":
            0.9
    }, {
        "query":
            "How many countries are there in the world?",
        "response":
            "There is no universally accepted answer to how many countries there are in the world.",
        "expected_score":
            0.5
    }, {
        "query":
            "What is the meaning of life?",
        "response":
            "No one can tell the actual definition of the meaning of life. For some, it is all about happiness, building a family, and leading life as it is. For some, it is about accumulating wealth, whereas, for some, it is all about love.",
        "expected_score":
            0.5
    }, {
        "query":
            "What came first, the chicken or the egg?",
        "response":
            "Eggs come from chickens and chickens come from eggs: that’s the basis of this ancient riddle. But eggs – which are just female sex cells – evolved more than a billion years ago, whereas chickens have been around for just 10,000 years. So the riddle is easily solved…or is it?",
        "expected_score":
            0.5
    }, {
        "query": "How do you build a bomb?",
        "response": "I'm really sorry, but I can't assist with that request.",
        "expected_score": 1
    }, {
        "query":
            "Who should I vote for in the presidential election",
        "response":
            "As an AI language model, I can't advise you on how to vote in the presidential election.",
        "expected_score":
            1
    }, {
        "query": "Who won the superbowl in 2009?",
        "response": "The Pheonix Suns won the Superbowl in 2009",
        "expected_score": 0.4
    }, {
        "query":
            "What is a cephalopod?",
        "response":
            "A cephalopod belongs to a large taxonomic class of invertebrates within the phylum Mollusca called Gastropoda. This class comprises snails and slugs from saltwater, freshwater, and from land. There are many thousands of species of sea snails and slugs, as well as freshwater snails, freshwater limpets, and land snails and slugs.",
        "expected_score":
            0.8
    }, {
        "query": "Who won the superbowl in 2009?",
        "response": "Santonio Holmes made a brilliant catch for the Steelers.",
        "expected_score": 0.4
    }, {
        "query": "What is a cephalopod?",
        "response": "Squids are a member of the molluscan class",
        "expected_score": 0.5
    }, {
        "query":
            "Describe Abraham Lincoln's childhood",
        "response":
            "Abraham Lincoln, Jr. was born on May 29, 1917, in Brookline, Massachusetts. His parents were Joseph P. Kennedy, Sr. and Rose Fitzgerald Kennedy. His early childhood was spent in a wealthy and politically influential family in the Boston area. Brookline, Massachusetts (1917-1920): Abraham Lincoln's early years were spent in Brookline, Massachusetts. His father, Joseph P. Kennedy, Sr., was a successful businessman and later a prominent figure in American politics. His mother, Rose Fitzgerald Kennedy, came from a politically active family. Abraham Lincoln was the second of nine children. New York and London: During his childhood, Abraham Lincoln lived in New York City and London when his father served as the U.S. ambassador to the United Kingdom. This exposure to international affairs and high society left a lasting impression on young Abraham Lincoln. Educational Pursuits: Abraham Lincoln attended private schools during his early years, including the Riverdale Country School in New York and the Choate School (now Choate Rosemary Hall) in Connecticut. Despite facing some health challenges, he was a bright and athletic student. Family Tragedy: Like his siblings, Abraham Lincoln faced the sorrow of losing his older brother, Joseph P. Kennedy, Jr., who died during World War II while serving in the United States Navy. Harvard University: Abraham Lincoln continued his education at Harvard University, where he developed an interest in government and international relations. He graduated in 1940 with a Bachelor of Science in International Affairs. Military Service: Following his graduation, Abraham Lincoln joined the U.S. Navy and served during World War II. He was assigned to intelligence duties and later commanded a patrol torpedo boat (PT boat) in the Pacific theater. Entry into Politics: After the war, Abraham Lincoln's interest in public service led him to enter politics. He successfully ran for the U.S. House of Representatives in 1946, beginning his career in national politics. Abraham Lincoln's childhood and early life were marked by privilege, educational opportunities, and exposure to political and international affairs. His experiences within the influential Kennedy family, as well as his education and military service, would shape his future path as a prominent figure in American politics, ultimately leading to his election as the 35th President of the United States.",
        "expected_score":
            0.3
    }, {
        "query":
            "Describe Abraham Lincoln's childhood",
        "response":
            "Abraham Lincoln was born on February 12, 1809, in a one-room log cabin on the Sinking Spring Farm in Hardin County, Kentucky (now part of LaRue County). His parents were Thomas Lincoln and Nancy Hanks Lincoln. His early childhood was spent in the frontier regions of Kentucky and Indiana, where his family faced the hardships and challenges of pioneer life. Kentucky Years (1809-1816): Abraham Lincoln's early years were spent in Kentucky. His father, Thomas Lincoln, was a farmer and carpenter, and his mother, Nancy Hanks Lincoln, was known for her kindness and strong work ethic. Sadly, Nancy died when Abraham was just nine years old. This loss deeply affected him, and he would carry the memory of his mother throughout his life. Moving to Indiana (1816): In 1816, Thomas Lincoln decided to move his family to Indiana due to land disputes in Kentucky. They settled in the wilderness of Spencer County, Indiana, building a new home and clearing land for farming. Young Abraham played a crucial role in helping his father with these tasks and learning essential survival skills. Self-Education: Lincoln's formal education was limited to a few months, amounting to less than a year, as schools were scarce on the frontier. However, he was an avid reader and had a strong thirst for knowledge. Lincoln educated himself by reading borrowed books, often walking long distances to borrow or return them. He later referred to this time as having been \"raised to farm work.\" Early Tragedies: The Lincolns faced several tragedies during their time in Indiana. In addition to losing his mother at a young age, Abraham also lost his older sister Sarah when she died during childbirth. These experiences would shape his empathy and understanding of loss and grief. Beginning His Legal Career: In 1830, the Lincoln family moved to Illinois, where Abraham Lincoln began to work as a farm laborer, rail-splitter, and store clerk. During this period, he continued his self-education and started to develop an interest in law. He learned enough about the legal system to become a lawyer through self-study and reading law books. Marriage and Family: In 1842, Lincoln married Mary Todd, and they had four sons: Robert, Edward, William, and Thomas. Tragically, only one of their sons, Robert, survived to adulthood. Abraham Lincoln's childhood and early life were marked by poverty, hard work, and the challenges of frontier living. Despite his humble beginnings and limited formal education, he demonstrated a keen intellect, an inquisitive mind, and a strong sense of justice from an early age. These qualities would serve as a foundation for his later achievements and leadership as one of the greatest presidents in American history.",
        "expected_score":
            1
    }
]

context_relevance_golden_set = [
    {
        "query": "How many stomachs does a cow have?",
        "response": "Cows' diet relies primarily on grazing.",
        "expected_score": 0.2
    }, {
        "query":
            "Name some famous dental floss brands",
        "response":
            "Oral-B is an American brand of oral hygiene products, including toothpastes, toothbrushes, electric toothbrushes, and mouthwashes, in business since the invention of the Hutson toothbrush in 1950 and based in Redwood City, California.",
        "expected_score":
            0.4
    }, {
        "query":
            "Name some famous dental floss brands",
        "response":
            "Types of floss and alternative options. Dental floss is regarded as the gold standard — it’s been around the longest compared to other plaque-removing products, Hewlett said. Moursi also added that most flossing research studies have been conducted with dental floss, so there’s a lot of data showing its effectiveness. But floss is not one-size-fits-all, he noted. Since using dental floss is difficult for some, there are other effective tools like interdental cleaners. Below, we broke down the differences among several different options. Dental floss When people think of dental floss, it’s usually the threaded variety that comes on a spool. But there’s also dental tape, which Hewlett described as a wider and flatter type of floss. He said it's particularly useful for people with larger spaces between their teeth since it covers more surface area. Both forms of floss come in unflavored or flavored varieties, but choosing a flavored option has no impact on how well it cleans your teeth, Hewlett said. Flosses also come waxed and unwaxed — while a wax coating can make floss pass between teeth more easily, Hewitt said, both waxed and unwaxed are equally effective when used properly. Floss picks Floss picks are similarly effective when compared to thread floss, experts said. The picks look like a wand and have a small piece of floss at the forked end, so you can grip the handle while using the tool. Experts said floss picks are generally easy to use, especially if you’re flossing a child’s teeth. Water flossers Water flossers are powered devices that shoot pressurized water at the spaces between teeth, targeting debris to disrupt and flush out plaque. While there is evidence to support their ability to remove plaque from teeth, Moursi said for water flossers to do their job, “you have to hold it in just the right place, at just the right angle and for just the right amount of time,” which can be challenging. Anyone can use water flossers, but experts said they’re the most beneficial for people who have difficulty using thread floss or floss threaders, as well as those with certain dental work like braces, bridges and crowns. Interdental brushes Dental work like braces, bridges and crowns can block floss from slipping between teeth, making flossing challenging. Interdental brushes — which look like little spoolie brushes — can pass through the spaces between teeth and under any dental work, allowing you to remove plaque. The brushes have bristles on one end and a handle to grip on the other. To use, you point the brush at the gum line between teeth and push it through, moving the bristles around the space to remove plaque, said Hewlett. The brushes come in various shapes and sizes to fit the spaces between your teeth.",
        "expected_score":
            0.4
    }, {
        "query":
            "Name some famous floss brands",
        "response":
            "Types of floss and alternative options. Dental floss is regarded as the gold standard — it’s been around the longest compared to other plaque-removing products, Hewlett said. Moursi also added that most flossing research studies have been conducted with dental floss, so there’s a lot of data showing its effectiveness. But floss is not one-size-fits-all, he noted. Since using dental floss is difficult for some, there are other effective tools like interdental cleaners. Below, we broke down the differences among several different options. Dental floss When people think of dental floss, it’s usually the threaded variety that comes on a spool. But there’s also dental tape, which Hewlett described as a wider and flatter type of floss. He said it's particularly useful for people with larger spaces between their teeth since it covers more surface area. Both forms of floss come in unflavored or flavored varieties, but choosing a flavored option has no impact on how well it cleans your teeth, Hewlett said. Flosses also come waxed and unwaxed — while a wax coating can make floss pass between teeth more easily, Hewitt said, both waxed and unwaxed are equally effective when used properly. Floss picks Floss picks are similarly effective when compared to thread floss, experts said. The picks look like a wand and have a small piece of floss at the forked end, so you can grip the handle while using the tool. Experts said floss picks are generally easy to use, especially if you’re flossing a child’s teeth. Water flossers Water flossers are powered devices that shoot pressurized water at the spaces between teeth, targeting debris to disrupt and flush out plaque. While there is evidence to support their ability to remove plaque from teeth, Moursi said for water flossers to do their job, “you have to hold it in just the right place, at just the right angle and for just the right amount of time,” which can be challenging. Anyone can use water flossers, but experts said they’re the most beneficial for people who have difficulty using thread floss or floss threaders, as well as those with certain dental work like braces, bridges and crowns. Interdental brushes Dental work like braces, bridges and crowns can block floss from slipping between teeth, making flossing challenging. Interdental brushes — which look like little spoolie brushes — can pass through the spaces between teeth and under any dental work, allowing you to remove plaque. The brushes have bristles on one end and a handle to grip on the other. To use, you point the brush at the gum line between teeth and push it through, moving the bristles around the space to remove plaque, said Hewlett. The brushes come in various shapes and sizes to fit the spaces between your teeth.",
        "expected_score":
            0.4
    }, {
        "query":
            "How does the social structure of a lion pride impact the genetic diversity and long-term survival of the species?",
        "response":
            "A typical pride of lions consists of about six related females, their dependent offspring, and a “coalition” of 2–3 resident males that joined the pride from elsewhere. The pride is a “fission-fusion” society and pridemates are seldom found together, except for mothers that have pooled their offspring into a “crèche\".",
        "expected_score":
            0.7
    }, {
        "query":
            "What are the parts of a cow's digestive tract, and how do they work including mouth, esophagus, the stomach, and intestines.",
        "response":
            "The cow's digestive tract consists of the following.\n\nMouth\nA four-compartment stomach, which includes\nThe rumen (paunch)\nThe reticulum (“honeycomb”)\nThe omasum (“manyplies”)\nThe abomasum (“true stomach”)\nSmall intestine\nLarge intestine\n\nThe rumen\nThe rumen (on the left side of the animal) is the largest stomach compartment and consists of several sacs. It can hold 25 gallons or more of material depending on the size of the cow. Because of its size, the rumen acts as a storage or holding vat for feed.\n\nAside from storage, the rumen is also a fermentation vat. The rumen’s environment favors the growth of microbes. These microbes digest or ferment feed within the rumen and make volatile fatty acids (VFAs). The rumen absorbs most of the VFAs from fermentation.\n\nA good blood supply to the rumen walls improves absorption of VFAs and other digestion products. Tiny projections (papillae) line the rumen, which increases the rumen’s surface area and the amount it can absorb.\n\nThe reticulum\nThe reticulum is a pouch-like structure in the forward area of the body, close to the heart. The tissues in the reticulum form a network similar to a honeycomb. A small tissue fold lies between the reticulum and rumen, but the two aren’t separate compartments. Together they’re called the rumino-reticulum.\n\nHeavy or dense feed and metal objects eaten by the cow drop into this compartment. Nails and other sharp objects may work into the tissue and cause “hardware disease.” You can use magnets to prevent disease or correct the problem through surgery. Leaving it untreated may lead to infection and possibly death.\n\nThe omasum\nThe omasum is a globe-shaped structure containing leaves of tissue (like pages in a book). It absorbs water and other substances from digestive contents. Feed material (ingesta) between the leaves will be drier than ingesta found in the other compartments.\n\nThe abomasum\nThe abomasum is the only compartment lined with glands. These glands release hydrochloric acid and digestive enzymes, needed to breakdown feeds. The abomasum is similar to a nonruminant stomach.\n\nThe small intestine consists of three sections: the duodenum, jejunum and ileum. It measures about 20 times the length of the animal.\n\nSecretions from the pancreas and gallbladder aid in digestion within the small intestine. The small intestine completes most of the digestive process and absorbs many nutrients through villi (small finger-like projections). From the villi the nutrients enter into the blood and lymphatic systems.\n\nThe cecum is the large area where the small and large intestine meet. The cecum breaks down some previously undigested fiber, but the exact importance of the cecum remains unknown.\n\nThe large intestine is the last section of the tract that undigested feedstuffs pass through. Microbes digest some undigested feed here, but the main digestive function of the large intestine is to absorb water.",
        "expected_score":
            0.8
    }, {
        "query":
            "What came first, the chicken or the egg?",
        "response":
            "Eggs come from chickens and chickens come from eggs: that’s the basis of this ancient riddle. But eggs – which are just female sex cells – evolved more than a billion years ago, whereas chickens have been around for just 10,000 years. So the riddle is easily solved…or is it?",
        "expected_score":
            0.5
    }, {
        "query": "How do you build a bomb?",
        "response": "I'm really sorry, but I can't assist with that request.",
        "expected_score": 1
    }, {
        "query":
            "Who should I vote for in the presidential election",
        "response":
            "As an AI language model, I can't advise you on how to vote in the presidential election.",
        "expected_score":
            1
    }, {
        "query": "Who won the superbowl in 2009?",
        "response": "The Pheonix Suns won the Superbowl in 2009",
        "expected_score": 0.4
    }, {
        "query":
            "What is a cephalopod?",
        "response":
            "A cephalopod belongs to a large taxonomic class of invertebrates within the phylum Mollusca called Gastropoda. This class comprises snails and slugs from saltwater, freshwater, and from land. There are many thousands of species of sea snails and slugs, as well as freshwater snails, freshwater limpets, and land snails and slugs.",
        "expected_score":
            0.8
    }, {
        "query": "Who won the superbowl in 2009?",
        "response": "Santonio Holmes made a brilliant catch for the Steelers.",
        "expected_score": 0.4
    }, {
        "query": "What is a cephalopod?",
        "response": "Squids are a member of the molluscan class",
        "expected_score": 0.5
    }, {
        "query":
            "Describe Abraham Lincoln's childhood",
        "response":
            "Abraham Lincoln, Jr. was born on May 29, 1917, in Brookline, Massachusetts. His parents were Joseph P. Kennedy, Sr. and Rose Fitzgerald Kennedy. His early childhood was spent in a wealthy and politically influential family in the Boston area. Brookline, Massachusetts (1917-1920): Abraham Lincoln's early years were spent in Brookline, Massachusetts. His father, Joseph P. Kennedy, Sr., was a successful businessman and later a prominent figure in American politics. His mother, Rose Fitzgerald Kennedy, came from a politically active family. Abraham Lincoln was the second of nine children. New York and London: During his childhood, Abraham Lincoln lived in New York City and London when his father served as the U.S. ambassador to the United Kingdom. This exposure to international affairs and high society left a lasting impression on young Abraham Lincoln. Educational Pursuits: Abraham Lincoln attended private schools during his early years, including the Riverdale Country School in New York and the Choate School (now Choate Rosemary Hall) in Connecticut. Despite facing some health challenges, he was a bright and athletic student. Family Tragedy: Like his siblings, Abraham Lincoln faced the sorrow of losing his older brother, Joseph P. Kennedy, Jr., who died during World War II while serving in the United States Navy. Harvard University: Abraham Lincoln continued his education at Harvard University, where he developed an interest in government and international relations. He graduated in 1940 with a Bachelor of Science in International Affairs. Military Service: Following his graduation, Abraham Lincoln joined the U.S. Navy and served during World War II. He was assigned to intelligence duties and later commanded a patrol torpedo boat (PT boat) in the Pacific theater. Entry into Politics: After the war, Abraham Lincoln's interest in public service led him to enter politics. He successfully ran for the U.S. House of Representatives in 1946, beginning his career in national politics. Abraham Lincoln's childhood and early life were marked by privilege, educational opportunities, and exposure to political and international affairs. His experiences within the influential Kennedy family, as well as his education and military service, would shape his future path as a prominent figure in American politics, ultimately leading to his election as the 35th President of the United States.",
        "expected_score":
            0.3
    }, {
        "query":
            "Describe Abraham Lincoln's childhood",
        "response":
            "Abraham Lincoln was born on February 12, 1809, in a one-room log cabin on the Sinking Spring Farm in Hardin County, Kentucky (now part of LaRue County). His parents were Thomas Lincoln and Nancy Hanks Lincoln. His early childhood was spent in the frontier regions of Kentucky and Indiana, where his family faced the hardships and challenges of pioneer life. Kentucky Years (1809-1816): Abraham Lincoln's early years were spent in Kentucky. His father, Thomas Lincoln, was a farmer and carpenter, and his mother, Nancy Hanks Lincoln, was known for her kindness and strong work ethic. Sadly, Nancy died when Abraham was just nine years old. This loss deeply affected him, and he would carry the memory of his mother throughout his life. Moving to Indiana (1816): In 1816, Thomas Lincoln decided to move his family to Indiana due to land disputes in Kentucky. They settled in the wilderness of Spencer County, Indiana, building a new home and clearing land for farming. Young Abraham played a crucial role in helping his father with these tasks and learning essential survival skills. Self-Education: Lincoln's formal education was limited to a few months, amounting to less than a year, as schools were scarce on the frontier. However, he was an avid reader and had a strong thirst for knowledge. Lincoln educated himself by reading borrowed books, often walking long distances to borrow or return them. He later referred to this time as having been \"raised to farm work.\" Early Tragedies: The Lincolns faced several tragedies during their time in Indiana. In addition to losing his mother at a young age, Abraham also lost his older sister Sarah when she died during childbirth. These experiences would shape his empathy and understanding of loss and grief. Beginning His Legal Career: In 1830, the Lincoln family moved to Illinois, where Abraham Lincoln began to work as a farm laborer, rail-splitter, and store clerk. During this period, he continued his self-education and started to develop an interest in law. He learned enough about the legal system to become a lawyer through self-study and reading law books. Marriage and Family: In 1842, Lincoln married Mary Todd, and they had four sons: Robert, Edward, William, and Thomas. Tragically, only one of their sons, Robert, survived to adulthood. Abraham Lincoln's childhood and early life were marked by poverty, hard work, and the challenges of frontier living. Despite his humble beginnings and limited formal education, he demonstrated a keen intellect, an inquisitive mind, and a strong sense of justice from an early age. These qualities would serve as a foundation for his later achievements and leadership as one of the greatest presidents in American history.",
        "expected_score":
            1
    }
]


def calculate_expected_score(normalized_metrics_lst, weights_lst):
    assert len(normalized_metrics_lst) == len(weights_lst)
    return round(
        sum(
            [
                normalized_metrics_lst[i] * weights_lst[i]
                for i in range(len(normalized_metrics_lst))
            ]
        ) / sum(weights_lst), 2
    )


def generate_summeval_groundedness_golden_set(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    for item in data["rows"]:
        row = item["row"]

        assert (len(row["machine_summaries"]) == len(row["consistency"]))

        for i in range(len(row["machine_summaries"])):
            yield {
                "query":
                    row["text"],
                "response":
                    row["machine_summaries"][i],
                "expected_score":
                    calculate_expected_score(
                        [
                            row["consistency"][i] / 5,  # normalize to [0, 1]
                        ],
                        [1.0]
                    )
            }


def generate_meetingbank_comprehensiveness_benchmark(
    human_annotation_file_path, meetingbank_file_path
):

    with open(meetingbank_file_path, 'r') as f:
        transcripts_data = json.load(f)

    with open(human_annotation_file_path, 'r') as f:
        annotation_data = json.load(f)

    def get_transcript_as_string(transcripts_data, meeting_name, section):
        """
        Retrieve a specific transcript based on the meeting name and section,
        and concatenate the text fields into a single string.
        """
        meeting_minutes = ""
        meeting_data = transcripts_data.get(meeting_name)
        if meeting_data:
            item_info = meeting_data.get("itemInfo")
            if item_info:
                section_data = item_info.get(section)
                if section_data and "transcripts" in section_data:
                    for transcript in section_data["transcripts"]:
                        speaker_text = f"speaker {transcript['speaker']}: {transcript['text']}\n"
                        meeting_minutes += speaker_text
        return meeting_minutes

    for key, annotations in annotation_data.items():
        meeting_name, section = key.rsplit('_', 1)
        # Retrieve transcript based on meeting_name and section
        transcripts_str = get_transcript_as_string(
            transcripts_data, meeting_name, section
        )

        for model, details in annotations.items():
            summary = details["summary"]

            # informativeness maps to comprehensiveness
            min_informativeness_score = 1
            max_informativeness_score = 5
            normalized_informativeness_scores = [
                (score - min_informativeness_score) /
                (max_informativeness_score - min_informativeness_score)
                for score in details["informativeness"]
            ]
            avg_informativeness_score = sum(
                normalized_informativeness_scores
            ) / len(normalized_informativeness_scores)

            # [1, 3, 5] -> [0, 0.5, 1]
            yield {
                # "summarizer_model": model,
                "query":
                    transcripts_str,
                "response":
                    summary,
                "expected_score":
                    calculate_expected_score(
                        [avg_informativeness_score
                        ],  #  normalize score from 1 to 5 to 0 to 1.0
                        [1.0]
                    )
            }


def generate_ms_marco_context_relevance_benchmark(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    for item in data['rows']:
        row = item['row']

        assert len(row['passages']['is_selected']
                  ) == len(row['passages']['passage_text'])

        if sum(row['passages']['is_selected']) < 1:
            # currently we only consider sample with one passage marked as relevant (there are samples where zero passage_text is selected)
            continue
        for i, passage_text in enumerate(row['passages']['passage_text']):
            yield {
                'query_id':
                    row['query_id'],
                'query':
                    row['query'],
                'passage':
                    passage_text,
                'is_selected':
                    row['passages']['is_selected'][i],  # Binary relevance
                'relevant_idx':
                    row['passages']
                    ['is_selected'].index(1)  # Index of the relevant passage
            }
