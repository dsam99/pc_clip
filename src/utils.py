from torchvision.datasets import ImageFolder, Flowers102, INaturalist, Food101, Places365
from data.dataset import AwA2, get_class_labels

def get_extended_class_labels(dataset_name):

    original_class_labels = get_class_labels(dataset_name=dataset_name)
    filepath = "extended_classes/" + dataset_name + ".txt"

    with open(filepath, "r") as f:
        class_labels = f.readlines()
    extended = [s.strip() for s in class_labels]

    for i in range(len(extended)):
        if "iostream" in extended[i]:
            extended[i] = original_class_labels[i]
    return extended

def get_descriptions(dataset_name, llm="chatgpt"):

    if dataset_name == "cifar10":
        
        if llm == "chatgpt":
            return [
                ["Fuselage and wings shape.","Tail or stabilizers.","Engine or propellers.","Landing gear.","Windows or cockpit area."],
                ["Four wheels and tires.","Car body shape.","Headlights and taillights.","Windshield and windows.","Grille or bumper design."],
                ["Feathered body and wings.","Beak shape and color.","Different wing spans.","Perching or flight posture.","Distinctive color patterns."],
                ["Ears and whiskers.","Eye shape and color.","Body size and proportions.","Fur texture and pattern.","Different breeds or features.",],
                ["Antlers (if present).","Body shape and size.","Leg proportions and stance.","Ear shape and position.","Different species or gender."],
                ["Snout shape and size.","Ear shape and position.","Tail length and curl.","Fur color and texture.","Different breeds or features."],
                ["Webbed feet.","Skin texture and color.","Eye position and size.","Leg proportions and posture.","Distinctive body shape."],
                ["Mane and tail.","Leg proportions and stance.","Muzzle and nostrils.","Different breeds or features.","Body size and shape."],
                ["Hull shape and structure.","Deck features and details.","Number and positioning of masts.","Presence of smokestacks.","Waterline and overall size."],
                ["Cargo bed or trailer.","Cab shape and structure.","Wheel size and design.","Grille and headlights.","Different truck styles or types."]
            ]

        elif llm == "flan-t5":
            return []
    elif dataset_name == "cifar100":
        return []
    elif dataset_name == "eurosat":
        if llm == "chatgpt":
            return [
                ["Leaf shape/color/pattern", "Crop row spacing/arrangement", "Plant height/density", "Flowering/fruiting stage", "Surrounding agricultural infrastructure/context"],
                ["Dense tree canopy cover.", "Various tree trunk shapes/colors.", "Multi-layered vegetation.", "Diverse leaf shapes/textures/colors.", "Natural surroundings (rivers, mountains)."],
                ["Low height and compactness.", "Broad leaf shapes and colors.", "Absence of woody stems.", "Soft and flexible stems.", "Often found in open fields."],
                ["Wide and paved road surface.", "Road markings and lane dividers.", "Traffic signs and signals.", "Overpasses or interchanges.", "Presence of vehicles in motion."],
                ["Large factory structures and chimneys.", "Heavy machinery and equipment.", "Industrial pipelines and storage tanks.", "Cranes or gantries.", "Presence of warehouses or loading docks."],
                ["Open grassy landscape.","Grazing livestock or animals.","Fencing or boundary markers.","Absence of trees or forested areas.","Rolling or undulating terrain."],
                ["Organized rows or patterns.","Distinctive tree shapes or structures.","Perennial vegetation with year-round presence.","Pruned or trained plant formations.","Presence of trellises or support structures.",],
                ["Houses or residential buildings.","Neatly arranged streets and roads.","Presence of driveways or garages.","Residential landscaping or gardens.","Street lighting or residential amenities.",            ],
                ["Flowing water or moving currents.", "Riverbank vegetation or trees.", "Meandering or winding shape.", "Presence of bridges or crossings.", "Reflective surface or water ripples.", ],
                ["Vast expanse of water.","Horizon line and open view.","Wave action or water movement.","Coastal or lakeside features.","Absence of significant current.",]
            ]    
        elif llm == "flan-t5":
            return []
    elif dataset_name == "sun":
        return []
    elif dataset_name == "cub":
        return []

    elif dataset_name == "oven":
        return []

    elif dataset_name == "oven_sub":
        return [
            ["Delicate, feathery foliage","Cladophylls (modified leaf-like stems)","Bright green coloration","Fern-like appearance","Fine, needle-like branches"],
            ["Pale or white stems","Feathery, needle-like foliage","Cladophylls (modified leaf-like stems)","Similar growth habit to green asparagus","Edible and tender shoots",],
            ["Long and cylindrical shape","Crusty exterior","Soft and chewy interior","Distinctive diagonal scoring patterns","Light golden brown color",],
            ["Transparent glass or plastic material","Sealable lid or cap","Sterilization indicators or labels","Capacity or volume markings","Designed for holding liquids",],
            ["Various shapes and sizes","Crust texture and color","Internal crumb structure","Toppings or fillings present","Sliced or unsliced appearance",],
            ["Toasted or golden brown exterior","Crispy texture","Warm or steaming appearance","Melting butter or spread","Sliced or whole loaf",],
            ["Dense, clustered florets","Vibrant green color","Thick, sturdy stalk","Presence of small leaves","Tree-like or bushy appearance",],
            ["Small, round cabbage-like heads","Light or dark green color","Compact, tightly packed leaves","Stalk or stem attachment","Distinctive bitter taste when cooked"],
            ["Light golden brown color","Uniform shape and size","Smooth, even texture","Absence of toppings or fillings","Partially cooked or set dough",],
            ["Round or rectangular shape","Decorative frosting or icing","Moist and tender crumb","Presence of layers or tiers","Sliced or whole presentation",],
        ]

    elif dataset_name == "flowers_sub":
        return [
            ["Pink flower color","Rounded petal shape","Basal rosette leaf arrangement","Lance-shaped leaves","Symmetrical flower structure",            ],
            ["Hard, leathery leaves","Compact growth habit","Unique flower shape","Distinctive coloration/pattern on petals","Presence of pseudobulbs",],
            ["Bell-shaped flowers","Multiple colors available","Tubular flower structure","Alternating leaf arrangement","Prominent veining on leaves",],
            ["Climbing vine habit","Fragrant blossoms","Pea-like flower shape","Spiraling tendrils for climbing","Pinnately compound leaves",],
            ["Bright orange or yellow flowers","Compact and bushy growth habit","Double or single flower forms","Pinnate or bipinnate leaves","Strong, distinct fragrance",],
            ["Showy orange or red flowers","Dark spots or speckles on petals","Erect, tall stem","Whorled or alternate leaf arrangement","Long, narrow, lance-shaped leaves",],
            ["Large, round, flat flowers","White or pale-colored petals","Fringed or ruffled edges","Column-like central structure","Glossy, waxy texture"],
            ["Vibrant, unique flower shape","Brilliant orange and blue colors","Boat-shaped flower bracts","Tall, erect stems","Long, banana-like leaves",],
            ["Tall, upright flowering spikes","Hooded or helmet-shaped flowers","Deep purple or blue petals","Delicate, finely divided leaves","Toxicity to humans and animals",],
            ["Spherical flower heads","Silvery or blue coloration","Spiky, thistle-like appearance","Long, sturdy stems","Pinnately lobed leaves",]
        ]

def get_comparatives(dataset, ft=True):

    '''
    Function to get comparative differences between classes that are provided by ChatGPT
    
    ft - a boolean flag whether to get standard CLIPs comparatives or those for our finetuned checkpoint
    '''

    if dataset == "eurosat":

        if ft:
            return [
                [(6, 0), "PermanentCrop zones have more uniform, perennial patterns with distinctive rows or tree clusters, while AnnualCrop areas often have varying colors/shapes due to seasonal planting."],
                [(9, 6), "SeaLake appears as large, uniform water bodies; PermanentCrop shows organized, often green, rows or tree patterns on land."],
                [(5, 6), "Pasture shows as relatively smooth, monochromatic fields; PermanentCrop has structured rows or patterns, typically with varied green hues."],
                # reverse
                # [(0, 6), "AnnualCrop areas often have varying colors/shapes due to seasonal planting,  while PermanentCrop zones have more uniform, perennial patterns with distinctive rows or tree clusters."],
                # [(6, 9), "PermanentCrop shows organized, often green, rows or tree patterns on land; SeaLake appears as large, uniform water bodies."],
                # [(6, 5), "PermanentCrop has structured rows or patterns, typically with varied green hues; Pasture shows as relatively smooth, monochromatic fields."],
            ]
        else:
            return [
                [(9, 6), "SeaLake appears as large, uniform water bodies; PermanentCrop shows organized, often green, rows or tree patterns on land."],
                [(6, 0), "PermanentCrop zones have more uniform, perennial patterns with distinctive rows or tree clusters, while AnnualCrop areas often have varying colors/shapes due to seasonal planting."],
                [(5, 0), "Pasture is more uniform in color, often green; AnnualCrop has varied colors and geometric shapes due to different crops and harvesting times."],
                # reverse
                [(0, 6), "AnnualCrop areas often have varying colors/shapes due to seasonal planting,  while PermanentCrop zones have more uniform, perennial patterns with distinctive rows or tree clusters."],
                [(6, 9), "PermanentCrop shows organized, often green, rows or tree patterns on land; SeaLake appears as large, uniform water bodies."],
                [(0, 5), "AnnualCrop has varied colors and geometric shapes due to different crops and harvesting times; Pasture is more uniform in color, often green."],
            ]
    
    if dataset == "oven_sub":
        return [
            [(2, 5), "\"Baguettes\" typically show fresh, whole loaves of bread, while \"bread_reheat\" may display reheated or leftover slices of bread."],
            [(4, 5), "\"Bread\" typically shows freshly baked loaves, while \"bread_reheat\" may depict slices of bread being reheated or toasted."],
            [(5, 8), "\"Bread_reheat\" suggests reheating slices of bread, while \"buns_prebaked\" may show unbaked or partially baked bun-like products before they are fully cooked."],
        ]

    if dataset == "flowers_sub":
        return [
            [(2, 8), "\"Canterbury bells\" typically have bell-shaped flowers, while \"monkshood\" features tall spikes of helmet-shaped flowers with a unique appearance."],
            [(1, 6), "A \"hard-leaved pocket orchid\" has robust, stiff leaves, while a \"moon orchid\" has delicate, round, and moon-like flowers."],
        ]

    elif dataset == "flowers":
        if ft:
            return [
                [(50, 97), "Petunia flowers have funnel-shaped blooms, often with a broad range of colors; Mexican Petunia bears trumpet-shaped flowers, typically in violet or blue hues."],
                [(55, 58), "The 'Bishop of Llandaff' dahlia has dark foliage with bright red flowers; orange dahlias have green foliage with orange blooms, varying in shape."],
                [(74, 18), "Thorn apple has spiky seed pods and trumpet-shaped white or purple flowers; balloon flower features rounded buds that open into bell-shaped, often blue or purple, blooms."],
            ]
        else:
            return [
                [(50, 97), "Petunia flowers have funnel-shaped blooms, often with a broad range of colors; Mexican Petunia bears trumpet-shaped flowers, typically in violet or blue hues."],
                [(55, 58), "The 'Bishop of Llandaff' dahlia has dark foliage with bright red flowers; orange dahlias have green foliage with orange blooms, varying in shape."],
                [(36, 42), "Cape flower (Strelitzia) resembles a bird with pointy orange and blue petals; sword lily (Gladiolus) has tall spikes with trumpet-shaped flowers in various colors."],
            ]

    elif dataset == "oven":
        return [
            [(36, 21), "A reheat plate is a microwave-safe dish with food, while an empty oven has no dishes or food inside, just its interior."],
            [(28, 12), "Lasagna typically features layered pasta and cheese with visible tomato sauce, while a casserole can vary widely in ingredients and appearance."],
            [(8, 5), "\"Buns_prebaked\" likely shows unbaked dough for buns, while \"bread_reheat\" suggests reheating already baked bread, visually distinct states."],
        ]
    
    elif dataset == "cub":
        if ft:
            return [
                [(123, 125), "Le Conte's Sparrow has buff-orange facial patterns and a fine streaked back; Nelson's Sharp-tailed Sparrow shows a grayish face with a crisp, streaked, brown back."],
                [(21, 91), "The Chuck-will's-widow has mottled brown plumage for camouflage; the Nighthawk is slimmer, with barred patterns and a white throat patch, often seen in flight at dusk."],
                [(109, 102), "Geococcyx (Roadrunner) is larger with long legs, a crest, streaked plumage, and a long tail; Sayornis (Phoebe) is smaller, plumper, with a dusky gray-brown color and short tail."],
            ]
        else:
            return [
                [(123, 125), "Le Conte's Sparrow has buff-orange facial patterns and a fine streaked back; Nelson's Sharp-tailed Sparrow shows a grayish face with a crisp, streaked, brown back."],
                [(21, 91), "The Chuck-will's-widow has mottled brown plumage for camouflage; the Nighthawk is slimmer, with barred patterns and a white throat patch, often seen in flight at dusk."],
                [(109, 102), "Geococcyx (Roadrunner) is larger with long legs, a crest, streaked plumage, and a long tail; Sayornis (Phoebe) is smaller, plumper, with a dusky gray-brown color and short tail."],
            ]

    elif dataset == "sun":
        if ft:
            return [
                [(213, 214), "A kitchen is typically larger with full-sized appliances; a kitchenette is smaller, with compact appliances and limited space."],
                [(300, 51), "A scene restaurant typically showcases a thematic, visually striking ambiance; a bistro is more casual, intimate, often with simple, cozy decor."],
                [(48, 192), "A scene bedroom may reflect personal style with varied colors and decorations; a hotel room is typically more uniform and standardized in design."],
            ]
        else:
            return [
                [(48, 192), "A scene bedroom may reflect personal style with varied colors and decorations; a hotel room is typically more uniform and standardized in design."],
                [(34, 289), "A bar typically has a sleek, modern design with bright lighting, while a pub/indoor often features a cozy, traditional atmosphere with darker, warmer tones."],
                [(326, 256), "Skyscrapers are taller, with a more striking, slender silhouette, while office buildings are shorter, with a bulkier, more functional shape."]
            ]
    
    elif dataset == "cifar100":
        if ft:
            return [
                [(26, 45), "Crabs have a rounded, flat body with two claws, while lobsters have a long body, large claws, and a pronounced tail."],
                [(47, 52), "Maple trees have lobed leaves and winged seeds; oaks have broader, lobed leaves and acorns."],
                [(63, 74), "A porcupine has a bulky body and quills, while a shrew is small, mouse-like, with a pointed snout and no quills."],
            ]
        else:
            return [
                [(26, 45), "Crabs have a rounded, flat body with two claws, while lobsters have a long body, large claws, and a pronounced tail."],
                [(98, 35), "A woman typically appears mature with more defined features; a girl looks youthful with softer, less developed characteristics."],
                [(47, 52), "Maple trees have lobed leaves and winged seeds; oaks have broader, lobed leaves and acorns."],
            ]

    elif dataset == "food":
        if ft:
            return [
                [(29, 83), "A photo of cupcakes shows small, individual cakes often with decorative frosting, while a red velvet cake photo features a large, single layered cake with a deep red color and typically creamy white frosting."],
                [(93, 37), "A photo of a steak shows a thicker, varied cut, while filet mignon is a smaller, round, and tender cut."],
                [(79, 93), "A photo of prime rib shows a large, roasted rib cut, often bone-in, while a steak is a smaller, individual slice."]
            ]
        else:
            return [
                [(29, 83), "A photo of cupcakes shows small, individual cakes often with decorative frosting, while a red velvet cake photo features a large, single layered cake with a deep red color and typically creamy white frosting."],
                [(93, 37), "A photo of a steak shows a thicker, varied cut, while filet mignon is a smaller, round, and tender cut."],
                [(79, 93), "A photo of prime rib shows a large, roasted rib cut, often bone-in, while a steak is a smaller, individual slice."]
            ]

def convert_comparatives_dict(comparatives, num_class=10):
    '''
    Function to convert a list of (tuple), string into a disctionary based on tuple values
    '''

    comp_dict = {}
    for i in range(num_class):
        comp_dict[i] = {}

    for comp in comparatives:
        ind1, ind2 = comp[0]
        string = comp[1]

        comp_dict[ind1][ind2] = string
    return comp_dict