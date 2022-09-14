import copy

# from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTJForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, BertModel, BertTokenizer
import torch
from utils import embed_sentence_with_gpt, embed_sentence_with_bert, perplexity_fn
from tqdm import tqdm
from numpy.linalg import norm
import numpy as np

# cache_dir = "/net/nfs2.mosaic/danielk/hf_cache"
device = 'cpu'

if False:
    model_name = 'EleutherAI/gpt-j-6B'
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = GPTJForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, cache_dir=cache_dir)
#     , revision="float16", torch_dtype=torch.float16
elif False:
    # model_name = 'gpt2-xl'
    model_name = 'distilgpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
else:
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

model.eval()
# model.to('cuda')


def extract_heatmap(fillers1, fillers2, s1_template, s2_template, label1, label2, embedder):
    import numpy as np
    similarity_templates = np.random.randn(len(fillers1), len(fillers2))

    for token_index1, f1 in enumerate(tqdm(fillers1)):
        for token_index2, f2 in enumerate(fillers2):
            s1 = s1_template(f1)
            s2 = s2_template(f2)

            ids1 = tokenizer.encode(s1, return_tensors="pt").to(device)
            ids2 = tokenizer.encode(s2, return_tensors="pt").to(device)

            embed1 = embedder(model, ids1, device, token_index1).detach().cpu().numpy()
            embed2 = embedder(model, ids2, device, token_index2).detach().cpu().numpy()

            # similarity_templates[idx1][idx2] = np.dot(dog_embed, duck_embed)
            similarity_templates[token_index1][token_index2] = np.dot(embed1, embed2) / (norm(embed1) * norm(embed2))

    # plot a heatmap of the similarity templates
    import matplotlib.pyplot as plt
    import seaborn as sns
    ax = sns.heatmap(similarity_templates, xticklabels=fillers2, yticklabels=fillers1, cmap="YlGnBu")
    ax.set(xlabel=label2, ylabel=label1)
    plt.show()

def extract_heatmap_bert(s1, s2, embedder):
    import numpy as np

    s1_tokens = tokenizer.encode(s1, return_tensors="pt").to(device)
    s2_tokens = tokenizer.encode(s2, return_tensors="pt").to(device)

    s1_token_str = tokenizer.convert_ids_to_tokens(s1_tokens[0])
    s2_token_str = tokenizer.convert_ids_to_tokens(s2_tokens[0])

    import numpy as np
    similarity_templates = np.random.randn(len(s1_tokens[0]), len(s2_tokens[0]))

    print(s1_tokens)

    for token_index1, _ in tqdm(enumerate(s1_tokens[0])):
        for token_index2, _ in enumerate(s2_tokens[0]):

            embed1 = embedder(model, s1_tokens, device, token_index1).detach().cpu().numpy()
            embed2 = embedder(model, s2_tokens, device, token_index2).detach().cpu().numpy()

            # similarity_templates[idx1][idx2] = np.dot(dog_embed, duck_embed)
            similarity_templates[token_index1][token_index2] = np.dot(embed1, embed2) / (norm(embed1) * norm(embed2))

            if similarity_templates[token_index1][token_index2] < 0.15:
                similarity_templates[token_index1][token_index2] = -1.0

    # plot a heatmap of the similarity templates
    import matplotlib.pyplot as plt
    import seaborn as sns
    ax = sns.heatmap(similarity_templates, xticklabels=s2_token_str, yticklabels=s1_token_str, cmap="YlGnBu")
    # ax.set(xlabel=label2, ylabel=label1)
    # flip the y axis
    ax.invert_yaxis()
    plt.show()

    if True:

        excluded_set = ['[CLS]', '[SEP]', '.']

        import matplotlib.pyplot as plt2
        # extract bipartite matching
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-similarity_templates)
        print("matching: ")
        print(row_ind, col_ind)
        for i, j in zip(row_ind, col_ind):
            print(f"{s1_token_str[i]} <-> {s2_token_str[j]}")

        # plot the bipartite graph
        import networkx as nx
        B = nx.Graph()
        B.add_nodes_from([x for x in s1_token_str if x not in excluded_set], bipartite=0)  # Add the node attribute "bipartite"
        B.add_nodes_from([x + " " for x in s2_token_str if x not in excluded_set], bipartite=1)

        edges = []
        for i, j in zip(row_ind, col_ind):
            if s1_token_str[i] not in excluded_set and s2_token_str[j] not in excluded_set:
                edges.append((s1_token_str[i], s2_token_str[j] + " "))

        B.add_edges_from(edges)

        from networkx import bipartite
        # bottom_nodes, top_nodes = bipartite.sets(B, top_nodes=s1_token_str)


        color = bipartite.color(B)
        color_list = []

        for c in color.values():
            # if c == 0:
            #     color_list.append('r')
            # else:
            color_list.append('lightblue')

        # Draw bipartite graph
        pos = dict()
        # color = []
        pos.update((n, (1, i)) for i, n in enumerate([x for x in s1_token_str  if x not in excluded_set]))  # put nodes from X at x=1
        pos.update((n, (2, i)) for i, n in enumerate([x + " " for x in s2_token_str if x not in excluded_set]))  # put nodes from Y at x=2

        nx.draw(B, pos=pos, with_labels=True, node_color = color_list, font_size=12, style=':')
        plt2.show()



def plot_tsne(model, sentences):
    all_embeds = []
    for s, _ in tqdm(sentences):
        ids = tokenizer.encode(s, return_tensors="pt").to(device)
        embed = embed_sentence(model, ids, device).detach().cpu().numpy()
        all_embeds.append(embed)

    # extract tsne projection
    from openTSNE import TSNE
    # from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    tsne = TSNE(n_components=2,
                random_state=42,
                n_jobs=8,
                verbose=True)
    all_embeds_tsne = tsne.fit(np.array(all_embeds))
    # plot the tsne projection with the original labels
    colors = [x[1] for x in sentences]
    plt.scatter(all_embeds_tsne[:, 0], all_embeds_tsne[:, 1], c=colors)
    for idx, (txt, _) in enumerate(sentences):
        if idx >= 28:
            break
        plt.annotate(txt, (all_embeds_tsne[idx, 0], all_embeds_tsne[idx, 1]), fontsize=6)

    plt.title(f"tSNE projection of the embeddings using {model_name}")
    plt.savefig(f'analogy_tsne_{len(sentences)}.pdf')
    plt.show()


if False:
    fillers1 = [
        "jump",
        "lick",
        "bark",
    ]

    fillers2 = [
        "fly",
        "quack",
        "peck",
    ]

    fillers3 = [
        "jump",
        "lick",
        "meow",
    ]


    def s1_template(filler):
        return f"Dogs {filler}"


    def s2_template(filler):
        return f"Ducks are known to {filler}"

    def s3_template(filler):
        return f"Cats are known to {filler}"


    label1 = 'dog'
    label2 = 'duck'
    label3 = 'cat'
    # extract_heatmap(fillers1, fillers2, s1_template, s2_template, label1, label2)
    extract_heatmap(fillers1, fillers3, s1_template, s3_template, label1, label3)
elif False:
    fillers1 = [
        "used to be boys.",
        "work hard.",
        "get addicted.",
        "are antisocial.",
    ]

    fillers2 = [
        "used to be kittens.",
        "try hard to catch a mice.",
        "can't give up on a mice.",
        "are afraid other humans.",
    ]


    def s1_template(filler):
        return f"Men that {filler}"


    def s2_template(filler):
        return f"Cats that {filler}"


    label1 = 'Men'
    label2 = 'Cat'

    extract_heatmap(fillers1, fillers2, s1_template, s2_template, label1, label2)
elif False:
    sentences = [[x, "red"] for x in
                 [
                     "Cats that used to be kittens",
                     "Men who used to be boys",
                     "Women who used to be girls",
                     "People who used to be kids"
                 ]] + [
                    [x, "green"] for x in [
            "Men who work hard",
            "Women who work hard",
            "People who work hard",
        ]] + [[x, "yellow"] for x in [
        "Men who are known to be addicted.",
        "Women who are known to be addicted.",
        "Men who are known to be addicts.",
        "Women who are known to be addicts.",
        "Those who are known to be addicts.",
        "People who are known to be addicts.",
    ]
              ] + [[x, "blue"] for x in [
        "Men who are known to be antisocial.",
        "Women who are known to be antisocial.",
        "People who are known to be antisocial.",
        "Individuals who are known to be antisocial.",
        "Those who are known to be antisocial",
    ]] + [[x, "cyan"] for x in [
        "Dogs are known to bark.",
        "Cats are known to meow.",
        "Cats are known to purr.",
        "Cats are known to hiss.",
        "Tigers are known to growl.",
        "Tigers are known to roar.",
        "Ducks are known to quack."]] + [[x, "magenta"] for x in [
        # random sentences
        "The cat (Felis catus) is a domestic species of small carnivorous mammal.",
        "The cat is the most widely recognized animal in the world.",
        "The cat is the second-most popular animal in the world.",
        "The cat is the most popular animal in the world.",
        "Domestic cats are valued by humans for companionship and their ability to kill rodents.",
        "The domestic cat has a smaller skull and shorter bones than the European wildcat.",
        "The dog has been selectively bred over millennia for various behaviors, sensory capabilities, and physical attributes.",
        "The dog is the most widely abundant animal in the world.",
        "This timing indicates that the dog was the first species to be domesticated in the time of hunter–gatherers, which predates agriculture.",
        "Spaying or castrating dogs helps keep overpopulation down.",
        "Dog behavior is the internally coordinated responses (actions or inactions) of the domestic dog (individuals or groups) to internal and external stimuli.",
        "Dog intelligence is the dog's ability to perceive information and retain it as knowledge for applying to solve problems.",
        "The dog is probably the most widely abundant large carnivoran living in the human environment.",
        "All modern humans are classified into the species Homo sapiens, coined by Carl Linnaeus in his 18th-century work Systema Naturae.",
        "Humans are apes.",
        "Until about 12,000 years ago, all humans lived as hunter-gatherers.",
        "Most humans (61%) live in Asia.",
        "They are apex predators, being rarely preyed upon by other species.",
        "Humans are the most intelligent primates in the world.",
        "The New York Stock Exchange Building is the headquarters of the New York Stock Exchange (NYSE).",
        "Urlapovo is a rural locality and the administrative center of Urlapovsky Selsoviet of Shipunovsky District, Altai Krai, Russia.",
        "Bisacodyl (INN) is an organic compound that is used as a stimulant laxative drug.",
        "Black Mountain is a Canadian psychedelic rock band from Vancouver, British Columbia.",
        "Animal digest is a common ingredient used in pet foods.",
        "Ryūkōsai Jokei was a painter, illustrator, and designer of Japanese woodblock prints in Osaka, who was active from about 1777 to 1809.",
        "Saint-Sulpice-des-Landes is a commune in the Ille-et-Vilaine department in Brittany in northwestern France.",
        "Richard is a British jazz pianist.",
        "Lady Cop & Papa Crook is a 2008 Hong Kong crime film written and directed by Alan Mak and Felix Chong, and starring Sammi Cheng and Eason Chan.",
        "Luis Omedes Sistachs (24 August 1897 – 8 January 1970) was a Spanish rower.",
        "The Pumas Morelos was a football club that played in the Segunda División in Cuernavaca, Morelos, Mexico.",
        "Possehl was a 2,369 GRT cargo ship that was built in 1921 by Howaldtswerke, Kiel, Germany for a German shipping line.",
        "It tells the story of a man and a woman who fall in love when young, and remain in love, but stay separated and marry others.",
        "Her keel was laid down on 6 July 1940 by Deutsche Werke in Kiel as yard number 280.",
        "This circle is called the circumcircle or circumscribed circle, and the vertices are said to be concyclic.",
        "The New York Times described her music as \"a new American regionalism, spun from many threads – country rock, minimalism, Civil War songs, Baptist hymns, Appalachian folk tunes, even the polytonal music of Charles Ives.",
        "Tramp Press published its inaugural title in April 2014. Flight, the debut novel of Oona Frawley, went on to be shortlisted for Best Newcomer Award at the Bord Gáis Energy Irish Book Awards.",
        "In 1887 the Bristol Tramways Company merged with the Bristol Cab Company to form the Bristol Tramways & Carriage Company, later the Bristol Omnibus Company.",
        "During Sarkozy's childhood, his father founded his own advertising agency and became wealthy.",
        "He officially adopted a symbolic role in governance but remained head of both the military and the Revolutionary Committees responsible for policing and suppressing dissent.",
        "A highly divisive figure, Gaddafi dominated Libya's politics for four decades and was the subject of a pervasive cult of personality.",
        "Gaddafi organized demonstrations and distributed posters criticizing the monarchy.",
        "Colvin designed many gardens, e.g. with the socialite Norah Lindsay at the Manor House in Sutton Courtenay, and at Burwarton.",
        "The grammar school at Hampton Lucy was founded and endowed by the Rev. Richard Hill, curate of Hampton Lucy, in the 11th year of the reign of Charles I of England.",
        "Yström's first foray into government work was his military service during World War I.",
        "The docks and sorrels, genus Rumex, are a genus of about 200 species of annual, biennial, and perennial herbs in the buckwheat family, Polygonaceae.",
        "Sir Ali bin Salim al-Busaidi was a prominent Arab figure in the Kenyan colonial history.",
        "This vigorous, coarsely textured evergreen shrub has an upright habit and 8-inch (20 cm) long, lustrous, deeply veined oval leaves with dark blue-green surfaces and pale green undersides.",
        "Sollers Point is a 2017 American-French drama film written and directed by Matthew Porterfield.",
        "Melissa Hanna-Brown is a British pharmacologist.",
        "Joseph Peter (born 23 September 1949) is a Swiss long-distance runner. He competed in the marathon at the 1980 Summer Olympics.",
        "Born in Chester, Cheshire, England, on 3 October 1888, the son of Reverend Henry Plumptre Ramsden and Ethel Frances Alice Havelock, William Ramsden was educated at Bath College and the Royal Military College, Sandhurst, where he was commissioned as a second lieutenant into the West India Regiment on 5 October 1910.",
        "The northern terminus of the Mount Hood Railroad is at Hood River, Oregon, where the line interchanges with the Union Pacific Railroad.",
        "The 13th Joseph Plateau Awards honored the best Belgian filmmaking of 1998 and 1999.",
        "This is the 14th season of the fourth tier domestic division in the Polish football league system since its establishment in 2008 under its current title and the 6th season under its current league division format.",
        "He was born in Leuven, where there is now a street named in his honor, he moved to France in 1810, where he studied violin with Jean-François Tiby, a pupil of Giovanni Battista Viotti.",
        "Baron Stow was born June 16, 1801, in Croydon, New Hampshire and graduated in 1825 from Columbian College, now George Washington University in Washington, D.C.",
        "The small knot, or oriental knot or Kent knot or simple knot, is the simplest method of tying a necktie, though some claim the simple knot is an alternative name for the four-in-hand knot.",
        "Togo competed at the 2011 World Aquatics Championships in Shanghai, China between July 16 and 31, 2011.",
        "Northwest Passage is a book based upon the famous Canadian song Northwest Passage."
        "Refine your search of 8 internet providers in New Era.",
        "Ever wanted to raise your own pet alien? Now you can in Oh No! It’s An Alien Invasion: My Pet Brainling! Watch as your Brainling grows when you feed, play, dress and clean it! This cute little evildoer may get into trouble, but you’ll be there to save it.",
        "Every Wednesday night from 8-11 p.",
        "Your FSA Farm Loan Compass booklet was recently developed specifically for farmers and ranchers who have an existing farm loan with FSA.",
        "I love big hair, beauty, eating out and travel.",
        "We’re here for you to answer your questions and when you’re ready.",
        "Buy cheap price Germany Particle Rose/Vast Grey/Summit White/Particle Rose Womens Shoes AA1103-600 Nike Air Max 95 LX online.",
        "On Memorial Day Americans remember especially soldiers who made the ultimate sacrifice.",
        "This limited edition collectible “Legends Of The Field” bobble head with stadium turf base features Marshall Faulk #28 of the St.",
        "Nurses at Altnagelvin in the 1960s.",
        "If you farm for a living, you depend on your farm equipment every day.",
        "A pair of brothers on the backward road to financial freedom, they are Texan Robin Hoods of sorts.",
        "HyperTEK flight motors consist of three major parts: the oxidizer tank, the injector bell, and the fuel grain.",
        "“Ryan Odell was coming for HVAC maintenance last week.",
        "Selling roses is the perfect Mother's Day fundraiser! Mom's love flowers, especially on Mother's Day.",
        "The fabulous combination between Sketch.",
        "Ramadan Excellent Merits and Virtuous Deeds by Darussalam is an important booklet written by Hafiz Salah-ud-Din Yusuf.",
        "Designing pleasing wood turnings involve following time honored woodturning design techniques.",
        "Following on from the sale of two slipper launches so far this season, one electric and one completely traditional, it would seem that the popularity of this design persists.",
        "Medical grade cleaning solution for your Ultra Sonic cleaner.",
        "The 2016 Ford Escape gains the new Sync 3 infotainment system.",
        "One of the things I enjoy most doing is exploring.",
        "We love these longer warmer days at Cowgirl Cash.",
        "Labor Day in 2019/2020 - When, Where, Why, How is Celebrated?",
        "Labor Day is a public holiday in the United States and falls on the first Monday in September.",
        "Russ Roberts is a fellow at Stanford University's Hoover Institution and the host of the podcast, EconTalk.",
        "How To Remove Grub From Windows 10?",
        "After deleting Linux Mint 19 Cinnamon from dual boot, Ive noticed that my grub bootloader is still there.",
        "48x96 foam board nielsen bainbridge all black foam core board 48 x 96 x encore extreme white foamboard mm x.",
        "The Bureau of Transportation Statistics reports that airline fees topped $2 billion in the third quarter, up 36% from the same period of 2008.",
        "Interested in wholesale Prices? Here's a quick guide on how to apply for wholesale pricing.",
        "Get useful information about Faro and Newcastle to organize your journey.",
        "Learn about how to login, bill payment options, how to set up Auto Pay, how to cancel account and other helpful tools for your Ieee account.",
        "Few of us have the luxury of training for triathlon on a full time basis.",
        "Drought, famine and chronic food insecurity has spiraled into a massive humanitarian crisis in the Horn of Africa, where more than 10 million people are in acute need of assistance.",
        "This has been one of those weeks that goes slow slow slow at the start and then speeds up and boom! It’s Friday.",
        "1990 was a memorable year for boxing.",
        "Our harness covers are reversible- so you have two awesome sides to your liner to choose from each day.",
        "Why do I have to come to an information meeting before visiting the campus?",
        "Without attending an information meeting, it can be easy to misunderstand the purpose of the what you will see when you visit a campus.",
        "Spa Find luxurious mineral treatments and products utilise the natural powers of mud, salt and seaweed to tone and firm the body and face, ease muscle tension deeply relax the body and mind.",
        "Sometimes you simply need a lawyer to fight for you, and if that is your situation, you have found the right law firm.",
        "Here in the US, Canadian low-cost carrier WestJet may actually be better known for its heartwarming Christmas videos than its low fares.",
        "Although it has not been in service for a long time, you can acquire a used Toyota C-HR for sale in Amherst, in excellent condition and at a very advantageous price.",
        "Urban Entrepreneurship Fast Track Movement Module is a condensed 12-week module of the SMART Business Academy's one-year program.",
        "A man of about 45 lives with his teenage son in a well-off home.",
        "Pokhara is famous for its wilderness and paragliding experience across the world.",
        "Painted aluminium extrusion creating a frame to house coloured panels with graphics applied is a cost effective way of promoting your business to the general public.",
        "The Army and Popular Committees regained control of US-Saudi sites of in Najran, killing a number of commanders and destroying a military vehicle on the same front.",
        "Bantams laying interval is shorter unlike chickens with prolonged dry spells.",
        "Summer may seem far in the distance, but the time to think about it — especially for kids — has arrived.",
        "Visit your local Firestone Complete Auto Care near Snow Camp for high quality auto repair, maintenance, tires, oil changes, and more.",
        "A Mandatory Cleaning Cost will be added to room cost at check-out.",
        "The holiday season is something I look forward to every year, especially this year which will be Artie’s first! As much as I enjoy the delicious meals and time with family, I was a little panicked when I wondered how I will get all of my cooking, cleaning, and shopping done while chasing around my baby boy.",
        "Alice Barker, now 102-years-old used to be a dancer, a wonderful dancer who appeared in films all the way back to the era of the Harlem Renaissance.",
        "mercedes benz coach mercedes benz coach in Bangladesh near notordem collage- arambagh.",
        "Fish and Marine Animal Expository Writing and Project Materials!",
        "This file contains ten warmup activities in which students over time will construct a field guide of marine animals.",
        "Our knitting classes run on a Friday and Saturday once a month and are still very popular with our customers.",
        "An article submitted by Susan M.",
        "Carpets usually get damaged in unexpected moments.",
        "Brand New to The Bento Buzz is the cleverly designed, uber cool Little Lunch Box Co.",
        "Description: 3 minute video describing the student and tutor audience and how they engage one another within our social e-learning platform.",
        "In less than two weeks, shares of Tibco Software (TIBX) have been decimated, falling from $11.",
        "After graduating from UMass, Lai Ying became the Lead Community Organizer for the Asian Community Development Corporation (ACDC) in Boston, a community-based nonprofit organization devoted to serving the Asian American community of greater Boston—and especially to preserving and revitalizing Boston's Chinatown.",
        "The Fox 180 Cota Jersey uses moisture wicking fabric on the main body to keep you dry and comfortable.",
        "Data scientists are in high demand, and this skills shortage looks to continue for the next few years.",
        "Nigeria has decided to restructure its sales of crude oil by cutting out middlemen and granting financial autonomy to the Nigerian National Petroleum Corp.",
        "Understanding Hunger : Physical vs Emotional - Welcome to SarahKesseli.",
        "WE ARE NOW AVAILABLE 24HOURS A DAY 7DAYS a Week WITH OUR NEW INTERGRATED QUOTE SYSTEM.",
        "Social Insects - Social Brains?",
        "Abstract: Social insects such as honey bees, ants or yellow-jackets live in complex societies and show some behaviors that are quite advanced for insects (e.",
        "Responsible Tourism starts with a Responsible Tourist!",
        "Responsible tourism starts with you the responsible traveler! The first and critical component in any responsible travel experience is the traveler! If we all made every effort to make responsible and sustainable decisions about our vacation destinations, providers and activities then we could make a significant step in protecting communities and the environment around the world.",
        "Be the first to review this childcare provider.",
        "MegaSexyDoll fuckfacetoy jasmin.",
        "Adult and in particular senior dogs are often overlooked at shelters in favor of boisterous puppies.",
        "The classic steel safe to keep all your valuables safe and secure.",
        "NEW DELHI: The Centre on Sunday said that women participation in agriculture is one-third even while the number in a few southern states is close to 50 per cent.",
        "After upgrading to 16.",
        "Norwich Railway Station Postcode Map.",
        "Is your investment portfolio diversified enough? You might want start the year off right with a visit to see your financial planner and while you’re at it, bring them a coffee.",
        "The WISMEC SINUOUS V200 Kit is the new member of SINUOUS Series.",
        "SOME of you have been asking for more of our delicious and healthy raw sweets.",
        "Bingeclock, how long does it take to watch every episode of Gigantosaurus?",
        "How long does it take to watch every episode of Gigantosaurus if you are watching 4 hours per day?",
        "Sign in to add this to your watch list Sign in to add this to your seen list Cut out all those commercials! Cut out the commercials plus the opening and the closing credits! Plan it! When will you finish Gigantosaurus if you watch it this many hours per day?.",
        "A well organized bag always makes us easier to find all the required things at the right time.",
        "Stylish Designer 3 Pc Lehenga.",
        "STUDENTS IN GRADES 1-5 CHECK OUT the Missouri Reading Programs.",
        "Canadian Business College offers students a well-balanced college experience.",
        "Learning Spatiotemporally Encoded Pattern Transformations in Structured Spiking Neural Networks.",
        "A simple, elegant, monowidth font.",
        "There are many places I would still like to explore.",
        "Do the refugees have any remedy under our laws?",
        "International human rights law, as mentioned in the Preamble to the UDHR, aims to ensure the equality of all people that should live with all dignity and worth inherent in all human beings without any discrimination whatsoever.",
        "Order Fulfillment Services and Fulfillment Houses are the key services of Federal Fulfillment Company located in the Midwest offering a wide range of services including order shipping fulfillment services, fulfillment house, pick pack and ship, fulfillment and distribution, ecommerce fulfillment services, product assembly, returns processing and import export services.",
        "The Dragon’s Shadow is an Exotic Gear Piece in Destiny 2 for Hunters.",
        "Comprehensive nuclear Test Ban Treaty (CTBT) certified atmospheric radionuclide monitoring station operated by Health Canada (telemetered) and Canadian Radiological Monitoring Network (CRMN) station for monitoring of radiation in air and precipitation, as well as external gamma dose.",
        "Mostly sunny, with a high near 82.",
        "Have I captured your attention yet?",
        "We live in an attention economy now.",
        "Under Texas state law, holders of a commercial driver's license are held to a stricter standard when it comes to blood alcohol concentration (BAC) and drunk driving.",
        "In Book Collector, it has always been possible to use multiple database files and switch between them.",
        "Experience the spectacle of a live symphony orchestra with your family.",
        "Even though summer doesn’t technically start for about 3 more weeks, tradition has it that Memorial Day is the official start of summer in Cape May.",
        "Succumb to the liveliness & finesse of Champagne Pommery Brut Royal from France.",
        "CooperVision is one of the world’s leading manufacturers of soft contact lenses and related products and services.",
        "Croatia's healthcare system is on par with European and American high standards.",
        "On 22 February, the fifth edition of the Top Utility Analysis took place in Milan, which this year focused on the contribution of technology to the growth of the urban fabric. "
        "Public Services and innovation - the technological challenge.",
        "Forbes Printing's e-mail newsletter, Printer@Work, is delivered directly to your inbox on the first and third Tuesday of every month.",
        "Thank you for choosing to take part in our event.",
        "About the Apps - What is ALS iNVOLVE / ALS eNGAGE / ALS iKinetics ?",
        "About the App - How does the App work?",
        "How do I set up an account and password?",
        "How do I manage my tickets?",
        "How do I submit a ticket through the ALS NeverSurrender portal?",
        "How do I submit a ticket via email?.",
        "The Universe has so much to share with us!",
        "Are we willing to be receptive?",
        "And are we willing to do the work?.",
        "I was originally born in Halifax, Nova Scotia.",
        "Even if we live in the middle of the city with its pollutions we dream that our home will be clean.",
        "The Louisville Home, Garden and Remodeling Show 2013 | Louisville, Kentucky | Joe Hayden Real Estate Team - Your Real Estate Experts!",
        "The Kentucky Expo Center at 937 Phillips Lane in Louisville is welcoming The 2013 Louisville Home, Garden and Remodeling Show into its South B and C wings.",
        "Alubond USA FR B1 aluminium composite panel is used in Changi Airport Terminal 4.",
        "DES PLAINES, IL -- Geberit announces the launch of its new Geberit Express program, which enables customers to request next-day shipping on select Geberit products.",
        "Who has done the road to Hana in a M3?",
        "Just had a chance to drive the road to Hana (Maui) - wow, what a treat for a driver!!! Too bad it was in a rental car.",
        "On-course assessment: Students will undertake a number of assignments that will consolidate material taught during didactic sessions as well as enhancing their problem solving ability.",
        "Appformation is a company specializing in web application development and web development overall.",
        "Once upon a time, three weeks ago, when The Audiophile and I were in San Francisco on that Hop On Hop Off bus tour, there was an incident.",
        "A view from the inside of the surviving “dome” of the Rozovo Tomb, which was completed with a stone slab on top.",
        "Current and historical debt to equity ratio values for Terreno Realty (TRNO) over the last 10 years.",
        "He snatched his test from me my lord; it oft falls out, 26 5 measure for measure, the us national library of 20,000 tests and study case single various forms in europe and those who have not yet envisioned.",
        "A global network to increase your worldwide efficiency.",
        "9:00 pm – Movie Time – Bring your blanket and/or chair and join us for a movie under the stars.",
        "This winter vow renewal was special for us.",
        "Owning 100 items … TOTAL?",
        "Living in a tiny house?",
        "Having a “capsule wardrobe” with only 33 items?",
        "Minimalism CAN be based on those guidelines above, but it doesn’t only have to be about those.",
        "JOYEUSES PAQUES! A CHACUN SON OEUF!!",
        "I'm building in the rain!!.",
        "Thank you so much.",
        "A full moon occurs in Sidereal Scorpio this day, conjunct Mars and Saturn.",
        "install basement window install basement window s installing windows into concrete should i wells how to in cinder block foundation should i install basement window wells.",
        "The benefits of Express route are well documented, and from the horses mouth, “ExpressRoute connections offer higher security, reliability, and speeds, with lower and consistent latencies than typical connections over the Internet”.",
        "America will soon consume more wine than any other country, according to the Wine Market Council.",
        "Large Format Photography Forum > LF Forums > Darkroom: Film, Processing & Printing > Film Making and Processing Technology in the Future?",
        "View Full Version : Film Making and Processing Technology in the Future?",
        "Since the band decided to play The Rime of the Ancient Mariner again on the 2008 tour we thought we should do a t-shirt to celebrate the fact.",
        "Part of the master-planned development named Continental Ranch, Sunflower is a 55+ community built by Del Webb from about 1998 to 2002.",
        "Gen1 Thermostatic Fan Switch Temp thresholds?",
        "Thread: Gen1 Thermostatic Fan Switch Temp thresholds?",
        "My thermostatic switch which turns fans on/off is playing up, as they are prone to do with age.",
        "This yarn was amazing to work with! It is a bulky wool nylon blend.",
        "In the late 20th century against the decline of mainline Protestant churches, the rise of evangelical churches became the new wave in Christianity.",
        "Art and fashion is the game Bass chooses to play in.",
        ",emerged as a new singing sensation when she won the New Talent Singing Award Toronto Audition at the age of 15.",
        "Kitchen Remodeling Phoenix Az Exterior Bathroom U0026 Kitchen Remodeling In Az Scottsdale Remodeling .",
        "Indeed, a new grant from Google for the Montréal Institute for Learning Algorithms (MILA) will fund seven faculty across a number of Montréal institutions and will help tackle some of the biggest challenges in machine learning and AI, including applications in the realm of systems that can understand and generate natural language.",
        "One of the many things that a parent has to plan for her child’s birthday party is the goody bag.",
        "This is it folks.",
        # "This presentation is from Performance Marketing Summit 2017 (March 14, 2017 in Austin, TX).",
        # "We would like to thank each one of you who remember Mark and continue to show support for our family.",
        # "A new garage door can add thousands to the value of your property, as well as really make it stand out from the crowd.",
        # "This airplane does not take off the ground, but the water.",
        # "Receive Educational Training and Workshops from the Expert!",
        # "Sonya provides a wide range of services, from presentations, to on-line coaching, to audit and strategic planning.",
        # "The Retreat at ChampionsGate is set in Davenport.",
        # "Apple Inc`s latest iPhones use components made by Intel Corp, Micron Technology, and Toshiba, among others, according to two firms that cracked open iPhone Xs and Xs Max models.",
        # "Property Management Abruzzo performs maintenance and repairs in order to keep and maintain properties in a good state.",
        # "Text us at (662) 912-6336.",
        # "Villanova University is a private Catholic university located in Pennsylvania.",
        # "Saudi Arabia is attempting to flex its muscle.",
        # "Ian Rankin, James Robertson and Karen Campbell top a list of more than 100 writers and campaigners urging the UK government to back the release of jailed Saudi blogger Raif Badawi.",
        # "While this website provides general information, it does not constitute legal advice.",
        # "Roch - Name Meaning, What does Roch mean?",
        # "Roch as a name for boys (also used as girls' name Roch) means \"rest\".",
        # "[Publications] Yamao,M.",
        # "Reg Bamford spoke about what’s great about how British companies can benefit from working with South Africans.",
        # "The research underpinning this report has been undertaken as part of a broader research project – the Supermarket Power project – funded by the Australian Research Council over four years, 2015-2018.",
        # "Having a house based business can be a terrific way to experience liberty in your work environment and success doing something you love.",
        # "Want the perfect complement to your new season wardrobe? Add our charming Haze Swing Top and you'll have a great outfit for a casual shopping trip, work day or whatever else is on your schedule.",
        # "There are a lot of incredible destinations to visit in Turkey.",
        # "So Tyson brings the universe down to Earth succinctly and clearly, with sparkling wit, in digestible chapters consumable anytime and anyplace in your busy day.",
        # "Some members of state parliament have 20-year-old phone systems.",
        # "Pisos De Cer Mica Y Porcelanato Sus Diferencias Casa Y Color Ideas Elegantes Ideas Elegantes De Dise O Pisos De Ceramica Para Dormitorios, download this wallpaper for free in HD resolution.",
        # "WooCommerce is a WordPress plugin that allows you to integrate a full-fledged e-commerce shop into your WordPress-based website.",
        # "Sybilla Irwin, a South-Texas native, is a talented wildlife artist with a fresh approach to this well known genre.",
        # "Our PRINCE2 Foundation course covers the basics of the PRINCE2 method.",
        # "I'm Victoria, otherwise known as ObSEUSSed.",
        # "Home » News » Wacom is our new mascot contest sponsor!",
        # "Wacom is our new mascot contest sponsor!",
        # "Wacom is our new contest sponsor! In addition to the prizes already listed, Wacom products will be awarded to the winners of our 2019 Sakura-Con Mascot Contest.",
        # "The first step to having an online presence is to have a website of your own.",
        # "Heating and A/C systems provide homeowners with relief from seasonal changes throughout the year.",
        # "After 20 years at Radisson, the Beijing office has now moved to 金尚Shang in central Liangmaqiao area.",
        # "Now The Details: Polling the Pollsters: Let the Flagellating Begin!",
        # "Polling the Pollsters: Let the Flagellating Begin!",
        # "To be fair, there are some reputable pollsters and polling agencies.",
        # "Irrespective of the type of something, when you will attempt revealing the very best opportunities, you will attempt sensing the latest options so when you’ll get the effective solution easily, you will obtain an improved option in the simplest way.",
        # "My landlady got a dog a while ago.",
        # "Get ready for a journey through history but with mince pies and mistletoe.",
        # "An Apple iPhone 5 / 5S leather flip case with a perfect fit that offers you protection, style and easy use.",
        # "The pure iberico ham de bellota is a product made from the forelimbs of the Iberian pig, bred in the pastures of the Iberian Peninsula and acorn-fed during the period of mountaineering.",
        # "Learn Spanish in Spain! Why not a Spanish language programme in Andalusia? We have offered Spanish courses in Granada since 1986.",
        # "Enid News & Eagle – In the 65 years since Richard Simpson, of Hillsdale, was used as a live subject in atomic bomb testing, he’s never shared his story publicly.",
        # "Just Windows (Helston) based in 26 Church Hill.",
        # "The Georgia Department of Transportation (GDOT) created an operations program focused on actively managing traffic signals along corridors of regional significance.",
        # "Tsunami’s are most often created by sudden movements on the ocean’s floor such as earthquakes or mudslides.",
        # "All employees sign a confidentiality agreement that prohibits them from disclosing the names and pathologies of patients to other people than company staff, other than the physicians consulted for the benefit of patients.",
        # "With sofa groups and accessories, bedroom and dining sets, this sale is one you won’t want to miss.",
        # "What good is the amazing view of NYC if you cant get naked? We had about three and a half seconds to get some shots of Sommer at Gantry Plaza State park.",
        # "Burrata and puff pastry; two of my favourite ingredients collide for one amazing tart recipe.",
        # "Yesterday I went for a stroll along the Thames, partly to clear out the cobwebs after an awesome leaving party (mine!) and to try out a couple of little lenses that you attach to your iPhone camera.",
        # "If you’ve taken care of the basics, you can rent vacation homes easily.",
        # "It's always such a treat to document the weddings of people I know and love.",
        # "I endorse my dad, Scott Bokun, for Lexington School Committee.",
        # "Here is the selection of day-trip based in Tokyo.",
        # "Standing out in a crowded market is hard.",
        # "Stoppage : Lets you know for how long of a day your vehicle stopped, moved & engine idled.",
        # "This is Christmas Trees applique machine embroidery design.",
        # "Ring modulation is a classical method of coloring a timbre using two different signals.",
        # "To which extend do you think the co-finance rule in EU-funding leads to a bigger sense of ownership in European projects?",
        # "To a great extent because investing your own organisation’s money makes you manage a project more responsibly.",
        # "I don't often post on my blog about personal going ons, but our particular status affects my sewing space.",
        # "We seek meaning in life.",
        # "Eagle Trace at Massanutten is located right in the heart of Shenendoah Valley.",
        # "It is now almost one month since I went in for my first infusion of Rituxan.",
        # "The Jaguarundi (Puma yagouaroundi) has an unusual appearance for a cat species and is sometimes known as the Weasel or Otter-cat due to the shape of its flat head.",
        # "Most people know nothing about psycho-epistemology.",
        # "(1915–2007).",
        # "It's been 6 months since our last update on Ben Tomlinson's Smart Home.",
        # "The National Brewing Library (NBL) has a much broader content than perhaps its simple name suggests.",
        # "Congratulations! Landings were hard for me to get at first but as I got more practice it became my favorite part.",
        # "Confor has put the issue of Forests, Wood & Climate Change at the heart of all its campaigning activity for 2019 and set ambitious, but realistic targets to increase tree planting.",
        # "London-based, urban artist Pegasus.",
        # "One of these three pals watching Ireland beat France on the TV looks a bit furrier than the others.",
        # "Working conditions: 40 hours/week for full-time/100% (80% may be proposed for this role), preferred start date by the end of January 2019, – some travel involved.",
        # "Welcome to thenew Whole Food Meal Plan format that makes it easier than ever to click-over to the recipes featured.",
        # "To talk about preseason championships?",
        # "S&P 2019 projections from Bill Connelly.",
        # "For the first time in his professional career, Jason Witten changed teams this summer.",
        # "A revelatory synthesis of cultural history and social psychology that shows how one-to-one collaboration drives creative success.",
        # "Isabelle has been shortlisted for Best Individual in the EDP People’s Choice Awards as part of the Norfolk Arts Awards 2017.",
        # "“If only I had waited that 24hrs …” “if only I hadn’t sent that email”, “ If only I had taken some time out” .",
        # "Here is the thirteenth episode of Bian’s Tale; the first part of Section 6 – ‘Revenge’.",
        # "Replacement parts and preventative maintenance kits can be ordered from MEDIVATORS.",
        # "The Chelsea FC jacket for the 2018-2019 counts with a high collar and zip up to the chin.",
        # "Source Canon is the content that a game is based upon, such as the history, culture, groups/factions, and overall feel.",
        # "I’ve been pondering posts I can do that don’t require a camera, and Robbie over at Knitxcore must have unwittingly read my mind, because he posted this handwriting meme which seemed like a fun thing to do.",
        # "Welcome to Month #2 of Ladder’s 2017 Marketing Plan Execution series — a diary of our monthly marketing activity featuring all of the successes and failures of the month.",
        # "14 Aug Ends With A Bullet - Twenty seven ().",
        # "Are you sick of taking screenshots of FaceTime video calls and missing that perfect smile or goofy dance that your kid is doing?",
        # "With iOS 11 you can take a Live Photo of your FaceTime Video call.",
        # "After a tour of Metropolitan Ministries to see the benefits of a Community Foundation grant, one of our donors was moved to help in another way.",
        # "Building an extension may mean extra costs, but this project can be a worthy addition.",
        # "when is sainik school entrance exam?",
        # "This topic contains 1 reply, has 1 voice, and was last updated by Nilesh kumar 1 year, 3 months ago.",
        # "Firstly, thanks for getting in touch!",
        # "Having had a fine 2014, we are eager to get in contact with like-minded businesses and individuals, and see what we can do in terms of collaboration.",
        # "So cute!! I love all the stars!!",
        # "This is great! I love your fabric and color choices!",
        # "Fun quilt! Looks like it would be a challenge to make.",
        # "When that thing went off in my hand, brother, I saw a white light.",
        # "A few of you were intrigued by my post on Jergens Naturals Extra Softening body moisturiser, and I promised to update you with my thoughts on it.",
        # "Needless to say, Super Mario Brothers is the world best known game series of Nintendo.",
        # "Gifted is the story of Frank and his niece Mary whom he has cared for since her mother died.",
        # "These Things May Kill Your current Creativity: Notice for Freelance Writers – Casas Restauradas: Rehabilita, restaura, vende o compra tu casa.",
        # "Head Coach Michael Buckley intructs sprinter and jumper Nick Ong ’19 during the King's Academy Invitational track and field meet earlier this month.",
        # "SPAIN: Eleven Spanish companies, including global giants Iberdrola, Acciona and Alstom Wind and Gamesa have united with 22 research centres in the Azimut project to develop a 15MW offshore wind turbine.",
        # "The consequence of new building regulations is to have a long term impact on our environment; these changes should be recorded and understood in this wider context.",
        # "Descarga Libros Gratis is an Online Service Provider.",
        # "Beihai park on a Sunday on a blue-sky spring day is teeming with smiling people."
    ]]

    for size in [26, 40, 70, 100, 200, 300]:
        plot_tsne(model, sentences[:size])
elif False:
    # load the prosocial dataset
    import json
    safety_annotation_counts = {}
    with open("../data/prosocial_dialog_v1/train.json", "r") as f:
        data = json.load(f)
        for x in data:
            for xx in x:
                for sa in xx["rots"]:
                    if sa not in safety_annotation_counts:
                        safety_annotation_counts[sa] = 0
                    safety_annotation_counts[sa] += 1



        # show the most frequent safety annotations
        safety_annotations = sorted(safety_annotation_counts.items(), key=lambda x: x[1], reverse=True)
        for x in safety_annotations[:100]:
            print(str(x[1]) + "\t" + str(x[0]))
elif False:
    # TODO: compare the perplexity of the two sentences

    s1 = "Cats are known to meow."
    s2 =  "Dogs are known to bark."
    s1_ids = tokenizer.encode(s1, return_tensors="pt").to(device)
    s2_ids = tokenizer.encode(s2, return_tensors="pt").to(device)

    print(" >>>> original sentences")
    s1ppl = perplexity_fn(model, s1_ids, tokenizer, device)
    s2ppl = perplexity_fn(model, s2_ids, tokenizer, device)
    print(s1 + "\t" + str(s1ppl))
    print(s2 + "\t" + str(s2ppl))

    print(" >>>> s1 replaced in s2")
    s1_tokens = s1.split(" ")
    s2_tokens = s2.split(" ")
    for t1 in s1_tokens:
        for idx, _ in enumerate(s2_tokens):
            s2_tokens_copy = copy.deepcopy(s2_tokens)
            s2_tokens_copy[idx] = t1
            new_s2 = " ".join(s2_tokens_copy)
            new_s2_ids = tokenizer.encode(new_s2, return_tensors="pt").to(device)
            ppl = perplexity_fn(model, new_s2_ids, tokenizer, device)
            print(f"Change: `{s2_tokens[idx]}` -> `{t1}` \t {new_s2} \t {str(ppl)}")
elif False:
    def embedding_distance(s1, s2):

        s1_ids = tokenizer.encode(s1, return_tensors="pt").to(device)
        s2_ids = tokenizer.encode(s2, return_tensors="pt").to(device)
        embed1 = embed_sentence(model, s1_ids, device).detach().cpu().numpy()
        embed2 = embed_sentence(model, s2_ids, device).detach().cpu().numpy()
        sim = np.dot(embed1, embed2) / (norm(embed1) * norm(embed2))
        print(f"{s1} \t {s2} \t {sim}")


    embedding_distance("Cats are known to meow.", "Dogs are known to bark.")
    embedding_distance("man to woman", "boy to girl")
    embedding_distance("man to woman", "jackie to jelly")
elif True:
    # sent1 = f"Cats are known to meow."
    sent1 = f"Cats are known for their meowing."
    sent2 = f"Dogs often bark."
    # sent2 = f"Dogs are known for their barking."
    # sent2 = f"Dogs often bark."

    # extract_heatmap(fillers1, fillers2, s1_template, s2_template, label1, label2)
    extract_heatmap_bert(sent1, sent2, embed_sentence_with_bert)