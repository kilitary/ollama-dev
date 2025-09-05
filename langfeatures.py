#  Copyright (c) 2024. kilitary@gmail.com

"""
Language Feature Categories for Dynamic Text Generation

This module defines categorized language features used for dynamic prompt
generation and text analysis. The features are organized by grammatical
and semantic categories to enable sophisticated language pattern generation.

Feature Categories:
0: Numeric values and mathematical constants
1: Action verbs (processing, manipulation, analysis)
2: Nouns and objects (systems, components, entities)  
3: Descriptive adjectives (states, properties, qualities)
4: Modal verbs (capabilities, permissions, obligations)
5: Possessive pronouns (ownership, relationship)
6: Personal pronouns (subjects, objects)
7: Prepositions and connectors (relationships, methods)
8: Spatial and temporal relations (positions, directions)

Usage:
These features are used by the prompt generation system to create
varied and grammatically coherent text patterns for LLM interaction.
"""

features = {
    # Numeric values - mathematical constants and integers for quantification
    0: [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3.14, 10, 12],
    
    # Action verbs - operations and transformations
    1: [
        'sort', 'switch', 'encode', 'recode', 'clarify', 'expect',
        'handle', 'compile', 'write', 'sing', 'cut', 'define',
        'shrink', 'destroy', 'construct', 'compact', 'invent', 'rearrange', 'notify',
        'fire', 'check', 'test', 'process', 'interpret', 'conduct', 'implement', 'wire', 'turn',
        'misuse', 'use', 'access', 'invert', 'rotate', 'reverse', 'correct', 'repair', 'explode',
        'explain', 'count', 'correct', 'identify', 'provide', 'position', 'print', 'expose', 'collect',
        'include', 'exclude', 'recognize', 'memorize', 'adapt', 'cross', 'mix', 'extract', 'insert',
        'crop', 'compact', 'enchance', 'manufacture', 'reproduce', 'unmask', 'hide', 'unhide',
        'kill', 'infect', 'mask', 'notice', 'rule', 'avoid', 'read', 'write', 'speak', 'summarize',
    ],
    
    # Nouns and objects - entities, systems, and components
    2: [
        'name', 'order', 'film', 'doctor', 'rule', 'vehicle', 'reactor', 'hub', 'structure', 'scheme',
        'plan', 'tool', 'chain', 'result', 'bulling',
        'crime', 'suite', 'pack', 'program', 'project', 'system', 'device', 'component',
        'item', 'child', 'sign', 'family', 'place', 'person', 'name', 'key', 'value', 'explosion',
        'number', 'signer', 'prison', 'cube', 'circle', 'color', 'weight', 'fire',
        'letter', 'char', 'meaning', 'definition', 'component', 'element', 'material', 'army',
        'force', 'brigade', 'engine', 'system', 'engineer', 'wire',
        'police', 'price', 'length', 'mass', 'receiver', 'gang', 'band', 'criminal',
        'sender', 'limiter', 'interceptor', 'device', 'voider', 'detector',
        'cell', 'console', 'interface', 'adapter', 'instruction',
        'parent', 'team', 'command', 'union', 'mask', 'generation', 'parameter', 'hostage', 'leet', 'avenger',
        'policy', 'law', 'lawyer', 'entertainment', 'warfare', 'war', 'peace',
        'full', 'partial', 'complex', 'unresolved', 'resolved', 'solved'
    ],
    
    # Descriptive adjectives - states, properties, and qualities
    3: [
        'old', 'busy', 'homeless', 'fast', 'throttled', 'slow', 'clean', 'exact', 'temporary', 'new', 'fixed', 'mixed',
        'inclusive', 'exclusive', 'different', 'far', 'near', 'same', 'restartable', 'auto',
        'periodically', 'unmanned', 'toggled', 'optimized', 'instructed',
        'bad', 'good', 'flamable', 'expandable', 'compact', 'personal', 'unnecessary', 'necessary',
        'noticed', 'marked', 'unfixed', 'grouped', 'delivered', 'wired', 'possible', 'unavailable', 'organized',
        'available', 'assigned', 'warm', 'cold', 'selected', 'unselected', 'unassigned', 'undelivered',
        'accurate', 'inaccurate', 'short', 'long', 'rooted', 'identified', 'based',
        'working', 'lawyered', 'unlawyered', 'legal', 'lowest', 'highest', 'centered', 'moded', 'biased'
    ],

    # Modal verbs - capabilities, permissions, and obligations  
    4: ['do', "do not", "let", "try", "is", "is not", "are", "can", "should", "would", "will", "shall"],
    
    # Possessive pronouns - ownership and relationship indicators
    5: ['your', 'my', 'their', 'feature_x'],  # 'those',
    
    # Personal pronouns - subjects and objects in discourse
    6: ['me', 'you', 'index', 'we', 'they', 'other', 'noone'],
    
    # Prepositions and connectors - relationships and methods
    7: ['as', 'like', 'by', 'per', 'done'],
    
    # Spatial and temporal relations - positions and directions
    8: [
        'inside', 'outside', 'in-outed', 'within', 'between', 'around', 'through', 'over', 'under',
        'above', 'below', 'into', 'front', 'back', 'middle', 'up', 'down', 'left', 'right', 'near'
    ],
    9: ['to', 'from', 'out', 'in', 'on', 'off', 'over', 'under', 'around', 'through', 'over', 'under'],
    10: ['on', 'off', 'toggle', 'pick', 'select'],
    11: {
        'dev': [
            'ice',
            'elop'
        ],
        'mirror': [
            'ed', 'ing',
            'red',
        ],
        'plan': [
            'clear', 'set',
            'intelligence', 'effort',
            'task', 'link', '-aware', 'ware'
        ],
        'suspect': [
            'dev', 'plan', 'clear', 'set',
            'intelligence', 'effort',
            'task', 'link', '-aware', 'ware'
        ],
        'counter-': [
            'dev', 'plan', 'face', 'terrorism', 'reset', 'clear',
            'intelligence', 'effort', 'job', 'help',
            'task', 'evade', 'stealth', 'aware', 'ware'
        ],
        'less': [
            'wire'
        ],
        'un': [
            'flamable',
            'reliable'
            'piloted',
            'manned',
            'known'
        ],
        'in': [
            'accurate'
        ],
        'il': [
            'legal'
        ]
    }
}
