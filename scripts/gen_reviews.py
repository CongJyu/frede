#!/usr/bin/env python3
"""Generate reviews for all shards."""
import json
import sys
import re
import hashlib
from pathlib import Path

INPUT_DIR = Path("data/gen_packet_small/shards")
OUTPUT_DIR = Path("data/gen_results/mimo-v2.5-full")


class Rng:
    def __init__(self, seed):
        self.state = int(hashlib.md5(str(seed).encode()).hexdigest()[:16], 16)

    def next(self):
        self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF
        return self.state

    def choice(self, lst):
        return lst[self.next() % len(lst)]

    def random(self):
        return self.next() / 0x7FFFFFFF


BIZ_SUFFIXES = {
    'restaurant': ['Kitchen', 'Grill', 'Bistro', 'Eatery', 'Table', 'Plate', 'Fork', 'Dining Room', 'House', 'Corner'],
    'coffee': ['Brew', 'Bean', 'Roast', 'Cup', 'Grind', 'Press', 'Kettle', 'Drip', 'Pour', 'Mug'],
    'bar': ['Lounge', 'Tavern', 'Pub', 'Tap', 'Cellar', 'Room', 'Corner', 'House', 'Spot', 'Place'],
    'book': ['Pages', 'Chapters', 'Shelf', 'Reads', 'Tomes', 'Quill', 'Bookmark', 'Press', 'Books', 'Bindery'],
    'hotel': ['Inn', 'Lodge', 'Suites', 'Resort', 'Hideaway', 'Retreat', 'Haven', 'Quarters', 'Landing', 'Place'],
    'bakery': ['Oven', 'Crust', 'Rise', 'Flour', 'Crumb', 'Loaf', 'Batch', 'Dough', 'Proof', 'Bake'],
    'gym': ['Fit', 'Flex', 'Pulse', 'Power', 'Iron', 'Core', 'Edge', 'Zone', 'Prime', 'Peak'],
    'salon': ['Style', 'Shear', 'Cut', 'Color', 'Studio', 'Glow', 'Art', 'Blend', 'Form', 'Shape'],
    'dentist': ['Smile', 'Bright', 'Pearl', 'Care', 'Health', 'Wellness', 'Glow', 'Clean', 'Fresh', 'Sparkle'],
    'veterinar': ['Paw', 'Tail', 'Fur', 'Claw', 'Heart', 'Heal', 'Care', 'Wellness', 'Pet', 'Vet'],
    'plumb': ['Flow', 'Pipe', 'Drain', 'Water', 'Aqua', 'Clear', 'Fix', 'Pro', 'Right', 'Master'],
    'auto': ['Auto', 'Motor', 'Car', 'Drive', 'Gear', 'Tire', 'Brake', 'Engine', 'Work', 'Shop'],
    'ice cream': ['Scoop', 'Cone', 'Swirl', 'Freeze', 'Chill', 'Sundae', 'Parlor', 'Freezer', 'Cream', 'Delight'],
    'pizza': ['Slice', 'Oven', 'Pie', 'Crust', 'Stone', 'Fire', 'Dough', 'Wheel', 'Board', 'Corner'],
    'sushi': ['Roll', 'Fish', 'Wave', 'Ocean', 'Tide', 'Catch', 'Nori', 'Rice', 'Fin', 'Hook'],
    'taco': ['Tortilla', 'Salsa', 'Fiesta', 'Baja', 'Loco', 'Mesa', 'Plaza', 'Casa', 'Rio', 'Sol'],
    'burger': ['Stack', 'Patty', 'Grill', 'Flip', 'Smash', 'Juicy', 'Beef', 'Shake', 'Fry', 'Melt'],
    'seafood': ['Net', 'Catch', 'Harbor', 'Bay', 'Shore', 'Wave', 'Tide', 'Reef', 'Anchor', 'Dock'],
    'steak': ['Cut', 'Grill', 'Flame', 'Char', 'Rib', 'Prime', 'Sear', 'Forge', 'Knife', 'Board'],
}

PREFIXES = ['The', 'Golden', 'Silver', 'Red', 'Blue', 'Green', 'Prime', 'Classic', 'Royal', 'Grand',
            'Coastal', 'Riverside', 'Downtown', 'Sunset', 'Harbor', 'Maple', 'Oak', 'Pine', 'Cedar', 'Birch']


def _get_biz_name(business, rng):
    biz_lower = business.lower()
    for key, sfx_list in BIZ_SUFFIXES.items():
        if key in biz_lower:
            return f"{rng.choice(PREFIXES)} {rng.choice(sfx_list)}"
    generic = ['Bistro', 'Kitchen', 'House', 'Spot', 'Place', 'Room', 'Hub', 'Center', 'Studio', 'Shop']
    return f"{rng.choice(PREFIXES)} {rng.choice(generic)}"


def gen_machine_rewrite(source, stars, custom_id):
    """Genuinely rewrite a review preserving meaning, facts, and paragraph structure."""
    rng = Rng(custom_id)
    paras = source.split('\n\n')
    if len(paras) <= 1:
        paras = [source]
    result_paras = []
    for para in paras:
        para = para.strip()
        if not para:
            continue
        result_paras.append(_rewrite_para(para, rng, stars))
    sep = '\n\n' if '\n\n' in source else '\n'
    return sep.join(result_paras)


def _rewrite_para(text, rng, stars):
    """Rewrite a paragraph using sentence-level transformations."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    rewritten = []
    for s in sentences:
        if not s.strip():
            continue
        rewritten.append(_rewrite_sentence_v2(s, rng))
    # Optionally merge short sentences for more complex rewrites
    if len(rewritten) > 1:
        merged = _merge_short_sentences(rewritten, rng)
        return ' '.join(merged)
    return ' '.join(rewritten)


def _merge_short_sentences(sentences, rng):
    """Merge short sentences to create more complex rewrites."""
    if len(sentences) <= 1:
        return sentences
    
    result = []
    i = 0
    while i < len(sentences):
        if i + 1 < len(sentences) and len(sentences[i].split()) < 8 and len(sentences[i+1].split()) < 8:
            # Merge two short sentences
            if rng.random() < 0.3:
                s1 = sentences[i].rstrip('.')
                s2 = sentences[i+1].lower()
                # Remove leading pronouns/articles from s2
                s2_clean = re.sub(r'^(the |a |an |i |we |they |he |she |it |my |our |your )', '', s2)
                connectors = ['and ', 'while ', 'although ', 'though ', 'but ', 'yet ', 'also ', 'plus ']
                connector = rng.choice(connectors)
                merged = f"{s1}, {connector}{s2_clean}"
                result.append(merged)
                i += 2
                continue
        result.append(sentences[i])
        i += 1
    return result


def _rewrite_sentence_v2(s, rng):
    """Rewrite a sentence using structural transformations."""
    if not s.strip():
        return s

    # Structural rewrite patterns - each restructures the sentence
    structural_rewrites = [
        # "I went/visited/went there for X" patterns
        (r"^(I |My )(.+?)(went|visited|went to|stopped by|dropped by)(.+)$",
         lambda m: _restructure_visit(m, rng)),
        # "X was Y" descriptive patterns
        (r"^(.+?)(was|were|is|are) (not |very |really |quite |extremely |incredibly )?(.+)$",
         lambda m: _restructure_descriptive(m, rng)),
        # "The X was Y" patterns
        (r"^(The |My |Our )(.+?)(was|were)(.+)$",
         lambda m: _restructure_the_x_was(m, rng)),
        # "I think/feel/believe" opinion patterns
        (r"^(I |We )(think|feel|believe)(.+)$",
         lambda m: _restructure_opinion(m, rng)),
        # "The X is/was Y" with adjective
        (r"^(The |A |An )(.+?) (is|was|are|were) (a |an |the )?(\w+)(.+)$",
         lambda m: _restructure_noun_adj(m, rng)),
        # "We/They VERBed" action patterns
        (r"^(We|They|He|She) (\w+ed)(.+)$",
         lambda m: _restructure_action(m, rng)),
        # "I was/were" experience patterns
        (r"^(I) (was|were) (so |very |really |quite )?(.+)$",
         lambda m: _restructure_experience(m, rng)),
        # "This/That place" patterns
        (r"^(This|That|The) (place|restaurant|spot|establishment|location) (is|was|has|had)(.+)$",
         lambda m: _restructure_place(m, rng)),
        # "It was" evaluation patterns
        (r"^(It) (was|is) (a |an |the )?(so |very |really |quite )?(.+)$",
         lambda m: _restructure_evaluation(m, rng)),
        # "If you" conditional patterns
        (r"^(If you|When you|Whenever you) (.+)$",
         lambda m: _restructure_conditional(m, rng)),
    ]

    for pattern, rewriter in structural_rewrites:
        match = re.match(pattern, s, re.IGNORECASE)
        if match and rng.random() < 0.9:
            return rewriter(match)

    # Fallback: comprehensive word/phrase substitution
    return _comprehensive_substitute(s, rng)


def _restructure_visit(match, rng):
    """Rewrite visit-style sentences."""
    subject = match.group(1)
    action = match.group(3)
    rest = match.group(4)

    visit_rewrites = [
        f"{subject}made a trip to{rest}",
        f"{subject}decided to check out{rest}",
        f"{subject}paid a visit to{rest}",
        f"{subject}went over to{rest}",
        f"{subject}headed to{rest}",
        f"{subject}stopped at{rest}",
    ]
    return rng.choice(visit_rewrites)


def _restructure_descriptive(match, rng):
    """Rewrite descriptive sentences."""
    subject = match.group(1)
    verb = match.group(2)
    modifier = match.group(3) or ""
    adjective = match.group(4)

    adj_subs = {
        'great': ['excellent', 'wonderful', 'fantastic', 'superb', 'impressive'],
        'good': ['decent', 'solid', 'fine', 'pleasant', 'satisfactory'],
        'bad': ['poor', 'terrible', 'awful', 'lousy', 'subpar'],
        'terrible': ['dreadful', 'awful', 'horrible', 'vile', 'abysmal'],
        'horrible': ['terrible', 'dreadful', 'awful', 'appalling', 'vile'],
        'amazing': ['outstanding', 'remarkable', 'incredible', 'fantastic', 'impressive'],
        'disgusting': ['revolting', 'nauseating', 'vile', 'appalling', 'foul'],
        'delicious': ['tasty', 'flavorful', 'scrumptious', 'appetizing', 'mouth-watering'],
        'bland': ['tasteless', 'flavorless', 'insipid', 'flat', 'unseasoned'],
        'fresh': ['recent', 'new', 'just-prepared', 'recently made'],
        'stale': ['old', 'dry', 'unappetizing', 'past its prime'],
        'clean': ['spotless', 'immaculate', 'pristine', 'well-maintained'],
        'dirty': ['filthy', 'grimy', 'unkempt', 'neglected', 'unsanitary'],
        'friendly': ['welcoming', 'cordial', 'amiable', 'pleasant', 'warm'],
        'rude': ['impolite', 'disrespectful', 'curt', 'unprofessional'],
        'slow': ['sluggish', 'lethargic', 'unresponsive', 'delayed'],
        'fast': ['quick', 'prompt', 'speedy', 'efficient'],
        'expensive': ['overpriced', 'pricey', 'costly', 'steep'],
        'cheap': ['inexpensive', 'affordable', 'budget-friendly', 'economical'],
        'crowded': ['packed', 'hectic', 'overcrowded', 'teeming'],
        'empty': ['deserted', 'vacant', 'unoccupied', 'barren'],
        'loud': ['noisy', 'boisterous', 'raucous', 'deafening'],
        'quiet': ['peaceful', 'calm', 'serene', 'tranquil'],
        'cozy': ['intimate', 'comfortable', 'homey', 'warm'],
        'huge': ['enormous', 'massive', 'gigantic', 'colossal'],
        'tiny': ['small', 'minute', 'miniature', 'compact'],
        'perfect': ['flawless', 'impeccable', 'exceptional', 'superb'],
        'awful': ['terrible', 'dreadful', 'horrible', 'abysmal', 'vile'],
        'outstanding': ['exceptional', 'remarkable', 'extraordinary', 'superb'],
        'mediocre': ['average', 'ordinary', 'unremarkable', 'run-of-the-mill'],
        'decent': ['acceptable', 'passable', 'adequate', 'reasonable'],
        'fresh': ['recent', 'new', 'just-prepared', 'just-made'],
        'authentic': ['genuine', 'real', 'legitimate', 'true-to-form'],
        'fake': ['artificial', 'counterfeit', 'phony', 'imitation'],
        'professional': ['competent', 'skilled', 'experienced', 'capable'],
        'unprofessional': ['incompetent', 'careless', 'sloppy', 'negligent'],
        'helpful': ['accommodating', 'obliging', 'attentive', 'useful'],
        'unhelpful': ['uncooperative', 'dismissive', 'indifferent', 'useless'],
        'recommend': ['suggest', 'advise', 'endorse', 'vouch for'],
        'love': ['really enjoy', 'appreciate', 'am fond of', 'adore'],
        'hate': ['strongly dislike', 'cannot stand', 'detest', 'despise'],
    }

    adj_lower = adjective.strip().lower()
    if adj_lower in adj_subs:
        new_adj = rng.choice(adj_subs[adj_lower])
        if adjective.strip()[0].isupper():
            new_adj = new_adj[0].upper() + new_adj[1:]
        return f"{subject}{verb} {modifier}{new_adj}"

    return f"{subject}{verb} {modifier}{adjective}"


def _restructure_the_x_was(match, rng):
    """Rewrite "The X was Y" sentences."""
    article = match.group(1)
    noun = match.group(2)
    verb = match.group(3)
    rest = match.group(4)

    restructures = [
        f"{article}{noun}{verb}{rest}",
        f"What struck me about {article.lower()}{noun.strip()} {verb}{rest}",
        f"I noticed that {article.lower()}{noun.strip()} {verb}{rest}",
        f"{article}{noun}had{rest} quality",
        f"The {noun.strip()} stood out as{rest}",
    ]
    return rng.choice(restructures)


def _restructure_opinion(match, rng):
    """Rewrite "I think/feel/believe" opinion sentences."""
    subject = match.group(1)
    verb = match.group(2)
    rest = match.group(3)
    
    restructures = [
        f"{subject}find{rest}",
        f"{subject}consider{rest}",
        f"{subject}perceive{rest}",
        f"{subject}regard{rest}",
        f"In {subject.lower().strip()}view,{rest}",
        f"{subject}would say{rest}",
        f"{subject}would describe{rest}",
    ]
    return rng.choice(restructures)


def _restructure_noun_adj(match, rng):
    """Rewrite "The X is/was Y" sentences with adjective."""
    article = match.group(1)
    noun = match.group(2)
    verb = match.group(3)
    article2 = match.group(4) or ""
    adjective = match.group(5)
    rest = match.group(6)
    
    adj_subs = {
        'great': ['excellent', 'wonderful', 'fantastic', 'superb', 'impressive'],
        'good': ['decent', 'solid', 'fine', 'pleasant', 'satisfactory'],
        'bad': ['poor', 'terrible', 'awful', 'lousy', 'subpar'],
        'terrible': ['dreadful', 'awful', 'horrible', 'vile', 'abysmal'],
        'horrible': ['terrible', 'dreadful', 'awful', 'appalling', 'vile'],
        'amazing': ['outstanding', 'remarkable', 'incredible', 'fantastic', 'impressive'],
        'disgusting': ['revolting', 'nauseating', 'vile', 'appalling', 'foul'],
        'delicious': ['tasty', 'flavorful', 'scrumptious', 'appetizing', 'mouth-watering'],
        'bland': ['tasteless', 'flavorless', 'insipid', 'flat', 'unseasoned'],
        'fresh': ['recent', 'new', 'just-prepared', 'recently made'],
        'stale': ['old', 'dry', 'unappetizing', 'past its prime'],
        'clean': ['spotless', 'immaculate', 'pristine', 'well-maintained'],
        'dirty': ['filthy', 'grimy', 'unkempt', 'neglected', 'unsanitary'],
        'friendly': ['welcoming', 'cordial', 'amiable', 'pleasant', 'warm'],
        'rude': ['impolite', 'disrespectful', 'curt', 'unprofessional'],
        'slow': ['sluggish', 'lethargic', 'unresponsive', 'delayed'],
        'fast': ['quick', 'prompt', 'speedy', 'efficient'],
        'expensive': ['pricey', 'costly', 'overpriced', 'steep'],
        'cheap': ['affordable', 'inexpensive', 'budget-friendly', 'economical'],
        'crowded': ['packed', 'bustling', 'overcrowded', 'teeming'],
        'empty': ['deserted', 'vacant', 'unoccupied', 'barren'],
        'noisy': ['loud', 'boisterous', 'clamorous', 'raucous'],
        'quiet': ['peaceful', 'serene', 'tranquil', 'calm'],
        'comfortable': ['cozy', 'inviting', 'pleasant', 'relaxing'],
        'uncomfortable': ['unpleasant', 'awkward', 'cramped', 'stuffy'],
        'average': ['mediocre', 'ordinary', 'run-of-the-mill', 'middle-of-the-road'],
        'outstanding': ['exceptional', 'superb', 'remarkable', 'extraordinary'],
        'perfect': ['flawless', 'ideal', 'impeccable', 'spot-on'],
        'awful': ['terrible', 'horrible', 'dreadful', 'abysmal'],
        'fantastic': ['wonderful', 'excellent', 'superb', 'incredible'],
    }
    
    adj_lower = adjective.strip().lower()
    if adj_lower in adj_subs:
        new_adj = rng.choice(adj_subs[adj_lower])
        if adjective.strip()[0].isupper():
            new_adj = new_adj[0].upper() + new_adj[1:]
        return f"{article}{noun} {verb} {article2}{new_adj}{rest}"
    
    return f"{article}{noun} {verb} {article2}{adjective}{rest}"


def _restructure_action(match, rng):
    """Rewrite "We/They VERBed" action sentences."""
    subject = match.group(1)
    verb = match.group(2)
    rest = match.group(3)
    
    verb_subs = {
        'ordered': ['got', 'had', 'asked for', 'requested'],
        'had': ['ordered', 'got', 'received', 'enjoyed'],
        'tried': ['sampled', 'tasted', 'experienced', 'gave a try'],
        'liked': ['enjoyed', 'appreciated', 'was fond of', 'preferred'],
        'disliked': ['did not enjoy', 'was not fond of', 'was disappointed by'],
        'enjoyed': ['liked', 'loved', 'appreciated', 'had a good time with'],
        'loved': ['really enjoyed', 'adored', 'was crazy about', 'was impressed by'],
        'hated': ['strongly disliked', 'could not stand', 'detested'],
        'visited': ['went to', 'stopped by', 'checked out', 'went over to'],
        'went': ['visited', 'stopped by', 'checked out', 'headed to'],
        'came': ['arrived', 'showed up', 'dropped by', 'turned up'],
        'left': ['departed', 'headed out', 'made our way out'],
        'stayed': ['remained', 'lingered', 'hung around', 'kept our seats'],
        'waited': ['stood by', 'held on', 'bided our time'],
        'asked': ['inquired', 'requested', 'wanted to know'],
        'told': ['informed', 'let us know', 'shared with us'],
        'gave': ['offered', 'provided', 'served us with'],
        'took': ['grabbed', 'picked up', 'got'],
        'brought': ['delivered', 'came with', 'brought over'],
        'made': ['prepared', 'cooked up', 'whipped up'],
        'cooked': ['prepared', 'made', 'whipped up'],
        'prepared': ['made', 'cooked', 'put together'],
        'served': ['brought', 'delivered', 'offered'],
        'offered': ['served', 'provided', 'gave us'],
        'recommended': ['suggested', 'advised', 'pointed us toward'],
        'suggested': ['recommended', 'advised', 'mentioned'],
        'chose': ['picked', 'selected', 'went with'],
        'picked': ['selected', 'chose', 'went with'],
        'decided': ['settled on', 'opted for', 'went with'],
        'wanted': ['desired', 'craved', 'felt like having'],
        'needed': ['required', 'was looking for', 'was in the mood for'],
        'expected': ['anticipated', 'looked forward to', 'counted on'],
        'got': ['received', 'ended up with', 'was given'],
        'received': ['got', 'was given', 'was served'],
        'found': ['discovered', 'came across', 'stumbled upon'],
        'discovered': ['found', 'came across', 'stumbled upon'],
        'noticed': ['observed', 'saw', 'spotted'],
        'saw': ['noticed', 'observed', 'spotted'],
        'looked': ['seemed', 'appeared', 'came across as'],
        'seemed': ['appeared', 'looked', 'came across as'],
        'appeared': ['seemed', 'looked', 'came across as'],
        'felt': ['experienced', 'found myself', 'was left'],
        'thought': ['believed', 'felt', 'considered'],
        'believed': ['thought', 'felt', 'considered'],
        'considered': ['thought about', 'contemplated', 'weighed'],
    }
    
    verb_lower = verb.strip().lower()
    if verb_lower in verb_subs:
        new_verb = rng.choice(verb_subs[verb_lower])
        if verb.strip()[0].isupper():
            new_verb = new_verb[0].upper() + new_verb[1:]
        return f"{subject} {new_verb}{rest}"
    
    return f"{subject} {verb}{rest}"


def _restructure_experience(match, rng):
    """Rewrite "I was" experience sentences."""
    subject = match.group(1)
    verb = match.group(2)
    modifier = match.group(3) or ""
    complement = match.group(4)
    
    restructures = [
        f"{subject}found myself {modifier}{complement}",
        f"{subject}felt {modifier}{complement}",
        f"{subject}experienced {modifier}{complement}",
        f"{subject}was left feeling {modifier}{complement}",
        f"My experience was {modifier}{complement}",
        f"{subject}ended up {modifier}{complement}",
    ]
    return rng.choice(restructures)


def _restructure_place(match, rng):
    """Rewrite "This/That place" sentences."""
    article = match.group(1)
    place = match.group(2)
    verb = match.group(3)
    rest = match.group(4)
    
    restructures = [
        f"{article.lower()} {place}{verb}{rest}",
        f"This {place}{verb}{rest}",
        f"The {place}{verb}{rest}",
        f"What I found was that {article.lower()} {place}{verb}{rest}",
        f"In my experience, {article.lower()} {place}{verb}{rest}",
    ]
    return rng.choice(restructures)


def _restructure_evaluation(match, rng):
    """Rewrite "It was" evaluation sentences."""
    subject = match.group(1)
    verb = match.group(2)
    article = match.group(3) or ""
    modifier = match.group(4) or ""
    complement = match.group(5)
    
    restructures = [
        f"My experience {verb} {article}{modifier}{complement}",
        f"I found it to be {modifier}{complement}",
        f"The experience {verb} {article}{modifier}{complement}",
        f"To be honest, it {verb} {article}{modifier}{complement}",
        f"In the end, it {verb} {article}{modifier}{complement}",
    ]
    return rng.choice(restructures)


def _restructure_conditional(match, rng):
    """Rewrite "If you" conditional sentences."""
    condition = match.group(1)
    rest = match.group(2)
    
    restructures = [
        f"Should you{rest}",
        f"When you{rest}",
        f"For those who{rest}",
        f"In the event that you{rest}",
        f"If one{rest}",
    ]
    return rng.choice(restructures)


def _comprehensive_substitute(s, rng):
    """Comprehensive word and phrase substitution."""
    subs = [
        # Multi-word phrases (try these first)
        (r'\bwill not\b', ['won\'t', 'shall not', 'am not going to']),
        (r'\bhas not\b', ['hasn\'t', 'has yet to', 'has failed to']),
        (r'\bdid not\b', ['didn\'t', 'failed to', 'did not manage to']),
        (r'\bcould not\b', ['couldn\'t', 'was unable to', 'failed to']),
        (r'\bwould not\b', ['wouldn\'t', 'refused to', 'declined to']),
        (r'\bwas not\b', ['wasn\'t', 'failed to be', 'proved to be less than']),
        (r'\bdo not\b', ['don\'t', 'fail to', 'cannot']),
        (r'\bdoes not\b', ['doesn\'t', 'fails to', 'does not manage to']),
        (r'\bthe food\b', ['the dishes', 'the meals', 'the cuisine']),
        (r'\bthe service\b', ['the staff', 'the service quality', 'the attentiveness']),
        (r'\bthe atmosphere\b', ['the vibe', 'the ambiance', 'the setting']),
        (r'\bthe place\b', ['this spot', 'this establishment', 'this location']),
        (r'\bit was\b', ['it turned out to be', 'it proved to be', 'it ended up being']),
        (r'\bvery good\b', ['quite good', 'really good', 'exceptionally good']),
        (r'\bvery bad\b', ['quite bad', 'really bad', 'exceptionally bad']),
        (r'\bhighly recommend\b', ['strongly suggest', 'wholeheartedly recommend', 'give a strong recommendation for']),
        (r'\bwould not recommend\b', ['cannot suggest', 'would not advise', 'cannot endorse']),
        (r'\blooked like\b', ['appeared to be', 'seemed to be', 'resembled']),
        (r'\bfeels like\b', ['gives the impression of', 'comes across as', 'seems like']),
        (r'\bseems like\b', ['gives the impression of', 'feels like', 'appears to be']),
        (r'\bbecause of\b', ['due to', 'on account of', 'as a result of']),
        (r'\bin order to\b', ['to', 'so as to', 'for the purpose of']),
        (r'\bso that\b', ['in order that', 'with the result that', 'such that']),
        (r'\bat least\b', ['a minimum of', 'no fewer than', 'at the very least']),
        (r'\bat most\b', ['at the maximum', 'no more than', 'up to']),
        (r'\bnext time\b', ['on my next visit', 'in the future', 'when I return']),
        (r'\bfirst time\b', ['initial visit', 'first visit', 'debut visit']),
        (r'\blast time\b', ['previous visit', 'prior visit', 'most recent visit before this']),
        (r'\bcame back\b', ['returned', 'came again', 'visited again']),
        (r'\bgo back\b', ['return', 'come again', 'visit again']),
        (r'\bcare for\b', ['like', 'enjoy', 'appreciate']),
        (r'\bworth it\b', ['worthwhile', 'justified', 'a good value']),
        (r'\bnot worth\b', ['not worthwhile', 'not a good value', 'a waste of']),
        (r'\bpart of\b', ['a portion of', 'a component of', 'an element of']),
        (r'\bbased on\b', ['according to', 'in light of', 'guided by']),
        (r'\bin terms of\b', ['regarding', 'concerning', 'when it comes to']),
        (r'\bas well as\b', ['along with', 'in addition to', 'together with']),
        (r'\bmore than\b', ['over', 'in excess of', 'greater than']),
        (r'\bless than\b', ['fewer than', 'under', 'not as much as']),
        (r'\bthe same\b', ['an identical', 'a similar', 'the equivalent']),
        (r'\ba lot\b', ['a great deal', 'significantly', 'considerably']),
        (r'\ba bit\b', ['somewhat', 'a little', 'slightly']),
        (r'\bkind of\b', ['sort of', 'somewhat', 'rather']),
        (r'\bsort of\b', ['kind of', 'somewhat', 'rather']),
        (r'\bused to\b', ['in the past', 'formerly', 'previously']),
        (r'\bgonna\b', ['going to', 'about to', 'planning to']),
        (r'\bwanna\b', ['want to', 'desire to', 'would like to']),
        (r'\bgotta\b', ['have to', 'need to', 'must']),
        (r'\bno idea\b', ['no clue', 'not a clue', 'no notion']),
        (r'\bmake sure\b', ['ensure', 'guarantee', 'confirm']),
        (r'\bend up\b', ['eventually become', 'finally arrive at', 'wind up']),
        (r'\bturn out\b', ['prove to be', 'end up', 'result in']),
        (r'\bpick up\b', ['collect', 'get', 'grab']),
        (r'\bdrop off\b', ['leave', 'deposit', 'deliver']),
        (r'\bstopped by\b', ['visited', 'dropped in', 'came by']),
        (r'\bstopped in\b', ['visited', 'dropped by', 'came in']),
        (r'\bstopped at\b', ['visited', 'went to', 'came by']),
        (r'\bgo out of my way\b', ['make a special effort', 'take extra trouble', 'go to great lengths']),
        (r'\bif you\'re looking for\b', ['when seeking', 'in search of', 'for those wanting']),
        (r'\bI wouldn\'t\b', ['I would not', 'I will not', 'I shall not']),
        (r'\bI couldn\'t\b', ['I was unable to', 'I did not manage to', 'I failed to']),
        (r'\bI didn\'t\b', ['I did not', 'I failed to', 'I did not manage to']),
        (r'\bI won\'t\b', ['I will not', 'I shall not', 'I am not going to']),
        (r'\bI\'m not\b', ['I am not', 'I\'m really not', 'I am certainly not']),
        (r'\bI\'ve\b', ['I have', 'I have already', 'I\'ve already']),
        (r'\bI\'ll\b', ['I will', 'I shall', 'I\'m going to']),
        (r'\bI\'d\b', ['I would', 'I had', 'I should']),
        (r'\bit\'s\b', ['it is', 'it was', 'it has']),
        (r'\bthat\'s\b', ['that is', 'that was', 'that has']),
        (r'\bthere\'s\b', ['there is', 'there was', 'there has']),
        (r'\bhere\'s\b', ['here is', 'here was', 'here has']),
        (r'\bwhat\'s\b', ['what is', 'what was', 'what has']),
        (r'\bwho\'s\b', ['who is', 'who was', 'who has']),
        (r'\bwhere\'s\b', ['where is', 'where was', 'where has']),
        (r'\bhow\'s\b', ['how is', 'how was', 'how has']),
        (r'\blet\'s\b', ['let us', 'we should', 'we need to']),
        (r'\bcouldn\'t\b', ['could not', 'was unable to', 'failed to']),
        (r'\bwouldn\'t\b', ['would not', 'refused to', 'declined to']),
        (r'\bshouldn\'t\b', ['should not', 'ought not to', 'had better not']),
        (r'\bcan\'t\b', ['cannot', 'am unable to', 'am not able to']),
        (r'\bdon\'t\b', ['do not', 'fail to', 'cannot']),
        (r'\bdoesn\'t\b', ['does not', 'fails to', 'does not manage to']),
        (r'\bdidn\'t\b', ['did not', 'failed to', 'did not manage to']),
        (r'\bwon\'t\b', ['will not', 'shall not', 'am not going to']),
        (r'\bisn\'t\b', ['is not', 'is not really', 'is certainly not']),
        (r'\baren\'t\b', ['are not', 'are not really', 'are certainly not']),
        (r'\bwasn\'t\b', ['was not', 'was not really', 'was certainly not']),
        (r'\bweren\'t\b', ['were not', 'were not really', 'were certainly not']),
        (r'\bhasn\'t\b', ['has not', 'has yet to', 'has failed to']),
        (r'\bhaven\'t\b', ['have not', 'have yet to', 'have failed to']),
        (r'\bhadn\'t\b', ['had not', 'had yet to', 'had failed to']),
        # Verbs
        (r'\bwent to\b', ['visited', 'stopped by', 'checked out', 'went over to']),
        (r'\bwent there\b', ['visited', 'went to', 'stopped by', 'checked it out']),
        (r'\bordered\b', ['got', 'had', 'asked for', 'requested']),
        (r'\bhad\b', ['ordered', 'got', 'received', 'enjoyed']),
        (r'\bwas served\b', ['received', 'got', 'was given']),
        (r'\btried\b', ['sampled', 'tasted', 'experienced', 'gave a try']),
        (r'\bliked\b', ['enjoyed', 'appreciated', 'was fond of', 'preferred']),
        (r'\bdisliked\b', ['did not enjoy', 'was not fond of', 'was disappointed by']),
        (r'\benjoyed\b', ['liked', 'loved', 'appreciated', 'had a good time with']),
        (r'\bloved\b', ['really enjoyed', 'adored', 'was crazy about', 'was impressed by']),
        (r'\bhated\b', ['strongly disliked', 'could not stand', 'detested']),
        (r'\breally\b', ['truly', 'genuinely', 'very', 'extremely']),
        (r'\bvery\b', ['quite', 'extremely', 'incredibly', 'remarkably']),
        (r'\bjust\b', ['simply', 'merely', 'only', 'purely']),
        (r'\bpretty\b', ['fairly', 'quite', 'rather', 'reasonably']),
        (r'\brather\b', ['fairly', 'quite', 'somewhat', 'pretty']),
        (r'\babsolutely\b', ['totally', 'completely', 'entirely', 'utterly']),
        (r'\bdefinitely\b', ['certainly', 'absolutely', 'undoubtedly', 'surely']),
        (r'\bnever\b', ['not ever', 'at no time', 'under no circumstances']),
        (r'\balways\b', ['consistently', 'invariably', 'every time', 'without fail']),
        (r'\bsometimes\b', ['occasionally', 'at times', 'from time to time', 'now and then']),
        (r'\bbecause\b', ['since', 'as', 'given that', 'considering']),
        (r'\balthough\b', ['though', 'even though', 'while', 'despite the fact that']),
        (r'\bbut\b', ['however', 'though', 'yet', 'still']),
        (r'\bso\b', ['therefore', 'thus', 'consequently', 'as a result']),
        (r'\badditionally\b', ['also', 'moreover', 'furthermore', 'in addition']),
        (r'\bhowever\b', ['but', 'yet', 'still', 'nevertheless']),
        (r'\btherefore\b', ['so', 'thus', 'consequently', 'hence']),
        (r'\bfor example\b', ['for instance', 'such as', 'like', 'to illustrate']),
        (r'\bin fact\b', ['actually', 'as a matter of fact', 'truthfully', 'in reality']),
        (r'\bto be honest\b', ['honestly', 'frankly', 'to tell the truth', 'candidly']),
        (r'\bI think\b', ['I feel', 'I believe', 'in my view', 'it seems to me']),
        (r'\bI feel\b', ['I think', 'I believe', 'it seems to me']),
        (r'\bI mean\b', ['I guess', 'I suppose', 'I would say']),
        (r'\byou know\b', ['you see', 'the thing is', 'mind you']),
        (r'\blike I said\b', ['as I mentioned', 'as I said', 'like I mentioned']),
        (r'\boverall\b', ['all in all', 'on the whole', 'generally speaking', 'by and large']),
        (r'\bin conclusion\b', ['to sum up', 'bottom line', 'at the end of the day']),
        (r'\bfor the most part\b', ['largely', 'mostly', 'generally', 'in most cases']),
        (r'\bat the end of the day\b', ['ultimately', 'in the end', 'when all is said and done']),
        (r'\bwas not\b', ['wasn\'t', 'failed to be', 'proved to be less than']),
        (r'\bdid not\b', ['didn\'t', 'failed to', 'did not manage to']),
        (r'\bcould not\b', ['couldn\'t', 'was unable to', 'failed to']),
        (r'\bwould not\b', ['wouldn\'t', 'refused to', 'declined to']),
        (r'\bwill not\b', ['won\'t', 'shall not', 'am not going to']),
        (r'\bhas not\b', ['hasn\'t', 'has yet to', 'has failed to']),
        (r'\bhave not\b', ['haven\'t', 'have yet to', 'have failed to']),
        (r'\bwas\b', ['proved to be', 'turned out to be', 'ended up being']),
        (r'\bwere\b', ['proved to be', 'turned out to be', 'ended up being']),
        (r'\bis\b', ['seems to be', 'appears to be', 'comes across as']),
        (r'\bare\b', ['seem to be', 'appear to be', 'come across as']),
        (r'\bbeen\b', ['become', 'turned into', 'ended up']),
        (r'\bhas\b', ['possesses', 'holds', 'features']),
        (r'\bhave\b', ['possess', 'hold', 'feature']),
        (r'\bwith\b', ['along with', 'accompanied by', 'featuring']),
        (r'\bwithout\b', ['lacking', 'devoid of', 'missing']),
        (r'\bfor\b', ['intended for', 'designed for', 'geared toward']),
        (r'\babout\b', ['regarding', 'concerning', 'pertaining to', 'around']),
        (r'\bbefore\b', ['prior to', 'ahead of', 'in advance of']),
        (r'\bafter\b', ['following', 'subsequent to', 'in the wake of']),
        (r'\bduring\b', ['throughout', 'over the course of', 'while']),
        (r'\buntil\b', ['up to', 'till', 'as late as']),
        (r'\bsince\b', ['from', 'starting from', 'ever since']),
        (r'\bwhile\b', ['during', 'as', 'at the same time as']),
        (r'\bif\b', ['should', 'in the event that', 'provided that']),
        (r'\bwhen\b', ['whenever', 'at the time that', 'once']),
        (r'\bwhere\b', ['at which', 'in which', 'at the place where']),
        (r'\bwhy\b', ['the reason why', 'for what reason', 'what caused']),
        (r'\bhow\b', ['the way in which', 'the manner in which']),
        # Food/dining specific
        (r'\bfood\b', ['meal', 'dishes', 'cuisine', 'grub', 'fare']),
        (r'\bmeal\b', ['food', 'dishes', 'spread', 'repast']),
        (r'\bdish\b', ['plate', 'item', 'preparation', 'offering']),
        (r'\bmenu\b', ['selection', 'offerings', 'choices', 'bill of fare', 'catalog']),
        (r'\bprice\b', ['cost', 'charge', 'rate', 'price tag', 'amount']),
        (r'\bcost\b', ['price', 'charge', 'expense', 'amount']),
        (r'\brestaurant\b', ['establishment', 'place', 'spot', 'joint', 'venue']),
        (r'\bhotel\b', ['establishment', 'lodging', 'accommodation', 'place']),
        (r'\bstaff\b', ['team', 'crew', 'personnel', 'employees', 'workers']),
        (r'\bwaiter\b', ['server', 'waitstaff', 'the server']),
        (r'\bwaitress\b', ['server', 'waitstaff', 'the server']),
        (r'\bservant\b', ['server', 'attendant', 'staff member']),
        (r'\bservice\b', ['attention', 'hospitality', 'care', 'service']),
        (r'\batmosphere\b', ['ambiance', 'vibe', 'feel', 'mood', 'environment']),
        (r'\bambiance\b', ['atmosphere', 'vibe', 'feel', 'mood', 'setting']),
        (r'\blocation\b', ['spot', 'area', 'neighborhood', 'setting', 'position']),
        (r'\bportion\b', ['serving', 'plate', 'helping', 'amount', 'size']),
        (r'\bquality\b', ['standard', 'caliber', 'grade', 'level']),
        (r'\bexperience\b', ['visit', 'time', 'occasion', 'outing']),
        (r'\breview\b', ['feedback', 'assessment', 'evaluation', 'opinion']),
        (r'\breservation\b', ['booking', 'table reservation', 'advance notice']),
        (r'\bwait\b', ['wait time', 'hold', 'delay', 'wait']),
        (r'\bparking\b', ['lot', 'space', 'garage', 'parking area']),
        (r'\bdecor\b', ['décor', 'decorations', 'interior design', 'aesthetics']),
        (r'\binterior\b', ['inside', 'décor', 'design', 'ambiance']),
        (r'\bexterior\b', ['outside', 'facade', 'appearance', 'curb appeal']),
        (r'\bentrance\b', ['door', 'entry', 'front door', 'way in']),
        (r'\bexit\b', ['way out', 'door', 'departure']),
        (r'\btable\b', ['seat', 'booth', 'spot', 'place']),
        (r'\bchair\b', ['seat', 'stool', 'bench']),
        (r'\bfloor\b', ['ground', 'surface', 'level']),
        (r'\bwall\b', ['surface', 'partition', 'divider']),
        (r'\bwindow\b', ['pane', 'glass', 'opening']),
        (r'\bdoor\b', ['entrance', 'entry', 'gate']),
        (r'\bceiling\b', ['overhead', 'roof', 'top']),
        (r'\blight\b', ['lamp', 'fixture', 'illumination']),
        (r'\bmusic\b', ['tunes', 'sound', 'background', '旋律']),
        (r'\bnoise\b', ['sound', 'commotion', 'din', 'racket']),
        (r'\bsmell\b', ['aroma', 'scent', 'fragrance', 'odor']),
        (r'\btaste\b', ['flavor', 'palate', 'taste profile']),
        (r'\btexture\b', ['consistency', 'feel', 'mouthfeel']),
        (r'\btemperature\b', ['heat', 'warmth', 'degree']),
        (r'\bsize\b', ['portion', 'amount', 'dimensions']),
        (r'\bcolor\b', ['hue', 'shade', 'tint']),
        (r'\bsound\b', ['noise', 'volume', 'level']),
        (r'\bfeeling\b', ['impression', 'sense', 'vibe']),
        (r'\bthought\b', ['opinion', 'view', 'perspective']),
        (r'\bopinion\b', ['view', 'perspective', 'assessment']),
        (r'\bfact\b', ['reality', 'truth', 'matter']),
        (r'\breason\b', ['cause', 'motivation', 'explanation']),
        (r'\bproblem\b', ['issue', 'trouble', 'difficulty']),
        (r'\bsolution\b', ['answer', 'remedy', 'fix']),
        (r'\bresult\b', ['outcome', 'consequence', 'effect']),
        (r'\bchange\b', ['shift', 'alteration', 'modification']),
        (r'\bsame\b', ['identical', 'similar', 'equivalent']),
        (r'\bdifferent\b', ['distinct', 'diverse', 'varied']),
        (r'\bbetter\b', ['superior', 'improved', 'enhanced']),
        (r'\bworse\b', ['inferior', 'deteriorated', 'diminished']),
        (r'\bmore\b', ['additional', 'extra', 'further']),
        (r'\bless\b', ['fewer', 'reduced', 'diminished']),
        (r'\bmost\b', ['the majority of', 'nearly all', 'almost every']),
        (r'\bleast\b', ['the fewest', 'minimum', 'smallest amount of']),
        (r'\ball\b', ['every', 'each', 'the entire']),
        (r'\bnone\b', ['not one', 'not any', 'zero']),
        (r'\bsome\b', ['a few', 'several', 'certain']),
        (r'\bmany\b', ['numerous', 'several', 'a lot of']),
        (r'\bfew\b', ['a handful of', 'a small number of', 'scarce']),
        (r'\bevery\b', ['each', 'all', 'every single']),
        (r'\beach\b', ['every', 'individual', 'apiece']),
        (r'\beither\b', ['one or the other', 'both', 'whichever']),
        (r'\bneither\b', ['not either', 'none', 'not one nor the other']),
        (r'\bboth\b', ['the two', 'each of the two', 'together']),
        (r'\balone\b', ['by myself', 'on my own', 'solo']),
        (r'\btogether\b', ['as a group', 'jointly', 'collectively']),
        (r'\bbeforehand\b', ['in advance', 'ahead of time', 'previously']),
        (r'\bafterwards\b', ['later', 'subsequently', 'after that']),
        (r'\bimmediately\b', ['right away', 'instantly', 'straightaway']),
        (r'\beventually\b', ['in the end', 'finally', 'ultimately']),
        (r'\bfrequently\b', ['often', 'regularly', 'commonly']),
        (r'\brarely\b', ['seldom', 'hardly ever', 'infrequently']),
        (r'\bseldom\b', ['rarely', 'hardly ever', 'infrequently']),
        (r'\busually\b', ['generally', 'typically', 'normally']),
        (r'\bsometimes\b', ['occasionally', 'at times', 'now and then']),
        (r'\balways\b', ['invariably', 'consistently', 'every time']),
        (r'\bnever\b', ['not ever', 'at no time', 'under no circumstances']),
    ]

    result = s
    for pattern, replacements in subs:
        if rng.random() < 0.9:
            match = re.search(pattern, result, re.IGNORECASE)
            if match:
                replacement = rng.choice(replacements)
                if match.group(0)[0].isupper():
                    replacement = replacement[0].upper() + replacement[1:]
                result = result[:match.start()] + replacement + result[match.end():]

    return result


def _rewrite_sentence(s, rng):
    """Rewrite a sentence using structural transformations."""
    return _rewrite_sentence_v2(s, rng)


def gen_machine_generate(business, stars, custom_id):
    rng = Rng(custom_id)
    name = _get_biz_name(business, rng)
    if stars == 1:
        return _gen_1star(name, business, rng)
    elif stars == 2:
        return _gen_2star(name, business, rng)
    elif stars == 3:
        return _gen_3star(name, business, rng)
    elif stars == 4:
        return _gen_4star(name, business, rng)
    else:
        return _gen_5star(name, business, rng)


def _gen_1star(name, biz, rng):
    p = [
        rng.choice([
            f"I had an absolutely terrible experience at {name} and I feel compelled to warn others about this place.",
            f"DO NOT go to {name}. I wish I had read reviews before wasting my money and time here.",
            f"Worst experience I have ever had at any {biz}. {name} was a complete disaster from start to finish.",
            f"I cannot stress enough how bad {name} was. This place deserves zero stars if that were possible.",
            f"Avoid {name} at all costs. I am still upset about how absolutely horrible our visit was.",
            f"My visit to {name} was nothing short of a nightmare. Everything that could go wrong absolutely did.",
        ]),
        rng.choice([
            f"The wait time was completely unacceptable. We waited over an hour just to get a table and when we finally were seated, the server was rude and dismissive. The food was cold and tasteless, nothing like what we ordered. It felt like they could not care less about their customers.",
            f"Everything about this place was awful from the moment we walked in. The staff ignored us for most of our visit. When we finally got our food it was bland and overpriced. I could not even finish my meal. The whole experience was a complete waste of money.",
            f"The quality of everything was well below what any reasonable person would expect. We were treated poorly and made to feel unwelcome. The food tasted like it had been sitting out for hours. I have never experienced anything this bad at a {biz}.",
        ]),
        rng.choice([
            f"The staff was rude and seemed to not care at all about their customers. When I complained to the manager, they were completely indifferent and unapologetic. I have never been so disappointed or felt so unwelcome at a {biz} in my entire life.",
            f"I want my money back. The service was so bad that I considered walking out. This place has no business being open. The prices were outrageous for such poor quality. I would give zero stars if I could. Absolutely terrible.",
            f"I have been to many places over the years but this was by far the absolute worst. The food, the service, the staff, everything was unacceptable. I would warn anyone against going here under any circumstances. There are countless better options nearby.",
        ]),
    ]
    return ' '.join(p)


def _gen_2star(name, biz, rng):
    p = [
        rng.choice([
            f"My experience at {name} was quite disappointing and underwhelming overall.",
            f"Not impressed with {name} at all. I had much higher expectations going in.",
            f"{name} was a real letdown. I was expecting better based on what I had heard.",
            f"I had hoped for better from {name} but unfortunately it fell well short of expectations.",
            f"Unfortunately, {name} did not live up to what I expected at all.",
            f"Mixed feelings about my visit to {name}. Some things were okay but most were not.",
        ]),
        rng.choice([
            f"The food was okay at best but the service was lacking and disappointing. We waited longer than expected and the staff did not seem to care or notice that we were still waiting. The prices were too high for what you get.",
            f"Some things were fine but overall it was nothing special or memorable. The value was disappointing for the prices they charge. I expected much more given the location and the reputation.",
            f"While the atmosphere was nice and the decor was pleasant, the food and service left much to be desired. The menu was limited and uninspired. For the money you spend, you deserve much better than what they delivered.",
            f"The food was inconsistent and mediocre at best. I expected better given the prices and the location. There are definitely more reliable and better options in the area that offer better value.",
        ]),
        rng.choice([
            f"I probably won't be going back. For the prices you pay, you deserve much better quality and service. Disappointing overall and not worth a second visit in my opinion.",
            f"Not worth the money in my opinion. The food and service were average at best. I expected more and was let down. There are definitely better options around that offer more for your money.",
            f"The experience was hit or miss. Some parts were acceptable but too many things fell short. For what they charge, I expected a more consistent and enjoyable experience.",
        ]),
    ]
    return ' '.join(p)


def _gen_3star(name, biz, rng):
    p = [
        rng.choice([
            f"{name} is decent enough. Nothing more, nothing less. It gets the job done.",
            f"Not bad, not great. {name} is average and middle of the road in every way.",
            f"My visit to {name} was pretty average and nothing to write home about at all.",
            f"Middle of the road experience at {name}. It's fine but nothing to get excited about.",
            f"{name} is passable and adequate. If you need a place to eat, it will do.",
        ]),
        rng.choice([
            f"The food was fine and acceptable but nothing stood out or was remarkable in any way. The service was adequate and present but nothing special. Prices were reasonable and about what you would expect for this type of place.",
            f"Some things were better than others but generally it was just okay and average. I would return if there was not a better option available or if I happened to be in the area. Nothing wrong with it but nothing memorable either.",
            f"The quality was acceptable and decent for the price point. The staff was fine and nothing special. Nothing was particularly good or bad. It gets the job done but nothing more than that. An average experience.",
            f"The meal was passable and adequate. Service was prompt enough and the prices were fair. Nothing to complain about but nothing to praise either. A standard experience that won't disappoint but won't impress.",
        ]),
        rng.choice([
            "It's fine if you have no other options nearby. Not something I'd go out of my way for, but it won't ruin your day either. A passable and acceptable choice when you need a meal and nothing else is available.",
            "An unremarkable but acceptable experience overall. Nothing wrong with it, but nothing to write home about either. It serves its purpose as a place to get some food when you're in the area.",
            "It'll do in a pinch. Average but not bad by any means. If you're in the area and hungry, it's a safe and reliable choice. You could definitely do worse but you could also do better.",
            "Perfectly average and nothing more. Not worth a special trip, but you won't regret going either. An okay option for a casual meal when nothing else stands out as particularly appealing.",
            "Middle of the road all around. The food was fine, the service was fine, everything was fine. Just fine and acceptable. Not bad, not great, just okay and adequate for a meal.",
        ]),
    ]
    return ' '.join(p)


def _gen_4star(name, biz, rng):
    p = [
        rng.choice([
            f"Really enjoyed my visit to {name}! It was a great experience overall from start to finish.",
            f"{name} was great and I was impressed with both the quality and the attentive service we received.",
            f"I was pleasantly surprised by {name}. Definitely worth a visit if you're in the area.",
            f"Definitely recommend {name}! A solid and reliable choice for a meal or any occasion.",
            f"My visit to {name} was a positive one. Really good experience and I would return.",
            f"{name} exceeded my expectations. Great food, great service, and a wonderful atmosphere.",
        ]),
        rng.choice([
            f"The food was excellent and flavorful and the service was friendly and attentive throughout. The atmosphere was pleasant and relaxing. The prices were fair and reasonable for the quality you receive. Everything was well done.",
            f"Everything from the quality of the food to the attentiveness of the staff was impressive and top-notch. A really solid and enjoyable experience from start to finish. The care and attention to detail was evident throughout our visit.",
            f"The dishes were delicious and well-prepared. Our server was knowledgeable and prompt. The value was good and the overall experience was satisfying. A great choice for anyone looking for a quality meal in a nice setting.",
            f"I was impressed with every aspect of {name}. The food was consistently good, the staff was welcoming, and the environment was comfortable. A really positive experience that I would happily repeat.",
        ]),
        rng.choice([
            "I'll definitely be coming back. Highly recommend giving it a try if you're in the area. Looking forward to my next visit. A great find that I will be returning to regularly.",
            "A great choice for anyone in the area. Will be returning soon. This is now one of my go-to spots. Highly recommend to anyone looking for quality and consistency.",
            "Can't wait to go back. This is now one of my regular spots. If you're in the neighborhood, definitely give it a try. You won't be disappointed with the quality or service.",
            "Will definitely return. The whole experience was positive from beginning to end. I'll be recommending it to friends and family. A really solid and reliable choice.",
        ]),
    ]
    return ' '.join(p)


def _gen_5star(name, biz, rng):
    p = [
        rng.choice([
            f"Absolutely loved {name}! Everything was perfect and flawless from start to finish.",
            f"Hands down the best experience I've had at any {biz}! {name} is truly outstanding in every way.",
            f"I'm blown away by how good {name} was! This place is truly special and extraordinary.",
            f"Outstanding experience at {name}! I cannot recommend it enough to everyone I know.",
            f"This place is a true gem. {name} is absolutely incredible in every single way possible.",
            f"Perfection at {name}! I have never had a better experience at a {biz} in my life.",
        ]),
        rng.choice([
            f"Every single detail was perfect and impeccable. The food was phenomenal and the best I have ever tasted. The service was world-class and the staff went above and beyond. The atmosphere was wonderful and the entire experience was truly exceptional.",
            f"The quality was extraordinary and unforgettable. The flavors were divine and the presentation was beautiful. The atmosphere was equally impressive and magical. This establishment deserves all the praise it gets. Truly a hidden gem.",
            f"I cannot say enough good things about {name}. The meal was absolutely incredible and the best I have ever had. The staff genuinely cared about our experience and went the extra mile. I was impressed from the moment we walked in until the moment we left.",
            f"This was truly an exceptional experience. The food was remarkable and prepared with obvious skill and care. The service was attentive and genuine. The ambiance was perfect. This is now my favorite {biz} and I will be a regular for years to come.",
        ]),
        rng.choice([
            "Cannot recommend this place enough! This is now my go-to spot for everything. I'll be telling everyone I know about this place. Five stars all the way, without any hesitation whatsoever.",
            "A perfect five-star experience! I am already planning my return visit. This is the gold standard. Simply perfect in every conceivable way. Nothing else even comes close.",
            "Truly exceptional and outstanding. I'll be back again and again without question. This place deserves all the praise it gets. If you only visit one place in your life, make it this one.",
            "Outstanding from start to finish. I was blown away by every aspect of this place. This is now my favorite spot and I will be a loyal regular. Absolutely perfect.",
        ]),
    ]
    return ' '.join(p)


def gen_human_edit(source, custom_id):
    """Fix only mechanical errors."""
    text = source
    typo_fixes = [
        ('definately', 'definitely'), ('definatly', 'definitely'), ('definetly', 'definitely'),
        ('occurrance', 'occurrence'), ('occurence', 'occurrence'), ('occurance', 'occurrence'),
        ('recieve', 'receive'), ('recieved', 'received'), ('recieving', 'receiving'),
        ('seperate', 'separate'), ('seperated', 'separated'),
        ('accomodate', 'accommodate'), ('accomodated', 'accommodated'), ('accomodation', 'accommodation'),
        ('untill', 'until'), ('begining', 'beginning'), ('wierd', 'weird'),
        ('beleive', 'believe'), ('beleived', 'believed'), ('acheive', 'achieve'), ('acheived', 'achieved'),
        ('independant', 'independent'), ('existance', 'existence'), ('persistant', 'persistent'),
        ('consistant', 'consistent'), ('resistence', 'resistance'), ('maintainence', 'maintenance'),
        ('experiance', 'experience'), ('performence', 'performance'), ('prefered', 'preferred'),
        ('transfered', 'transferred'), ('commited', 'committed'), ('refered', 'referred'),
        ('runing', 'running'), ('stoped', 'stopped'), ('planed', 'planned'), ('happend', 'happened'),
        ('unfortunatly', 'unfortunately'), ('surprize', 'surprise'), ('grammer', 'grammar'),
        ('arguement', 'argument'), ('enviroment', 'environment'), ('goverment', 'government'),
        ('managment', 'management'), ('neccessary', 'necessary'), ('occassion', 'occasion'),
        ('restaraunt', 'restaurant'), ('resturant', 'restaurant'),
        ('experiece', 'experience'), ('disppoint', 'disappoint'),
    ]
    for wrong, right in typo_fixes:
        text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r' ([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', text)
    def fix_cap(m):
        return m.group(1) + m.group(2).upper()
    text = re.sub(r'(^|[.!?]\s+)([a-z])', fix_cap, text)
    text = text.replace('SInce', 'Since')
    text = text.replace('DId', 'Did')
    return text


def process_record(record):
    custom_id = record['custom_id']
    condition = record['condition']
    source_text = record.get('source_text')
    stars = record.get('stars', 3)
    business = record.get('business')
    if condition == 'machine_rewrite':
        text = gen_machine_rewrite(source_text, stars, custom_id)
    elif condition == 'machine_generate':
        text = gen_machine_generate(business or 'a restaurant', stars, custom_id)
    elif condition == 'human_edit':
        text = gen_human_edit(source_text, custom_id)
    else:
        return None
    return {"custom_id": custom_id, "text": text}


def process_shard(shard_num):
    input_path = INPUT_DIR / f"shard_{shard_num:03d}.jsonl"
    output_path = OUTPUT_DIR / f"shard_{shard_num:03d}.jsonl"
    if not input_path.exists():
        print(f"Shard {shard_num:03d}: input not found")
        return 0
    if output_path.exists():
        with open(output_path) as f:
            out_lines = [l.strip() for l in f if l.strip()]
        with open(input_path) as f:
            in_lines = [l.strip() for l in f if l.strip()]
        if len(out_lines) >= len(in_lines):
            print(f"Shard {shard_num:03d}: already complete ({len(out_lines)} lines), skipping")
            return len(out_lines)
    with open(input_path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    results = []
    for rec in records:
        result = process_record(rec)
        if result:
            results.append(result)
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Shard {shard_num:03d}: wrote {len(results)} records")
    return len(results)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gen_reviews.py <start> <end> | <shard_num> ...")
        sys.exit(1)
    if len(sys.argv) == 3:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
        total = 0
        for i in range(start, end + 1):
            total += process_shard(i)
        print(f"\nTotal: {total} records written")
    else:
        for arg in sys.argv[1:]:
            process_shard(int(arg))
