
FORMAT = 'a tissue sample from {} affected by {}'
CELLTYPE_FORMAT = '{}, {}, and {}'
# from PathKEP
KEP_Templates = ['CLASSNAME.',
            'a photomicrograph showing CLASSNAME.',
            'a photomicrograph of CLASSNAME.',
            'an image of CLASSNAME.',
            'an image showing CLASSNAME.',
            'an example of CLASSNAME.',
            'CLASSNAME is shown.',
            'this is CLASSNAME.',
            'there is CLASSNAME.',
            'a histopathological image showing CLASSNAME.',
            'a histopathological image of CLASSNAME.',
            'a histopathological photograph of CLASSNAME.',
            'a histopathological photograph showing CLASSNAME.',
            'shows CLASSNAME.',
            'presence of CLASSNAME.',
            'CLASSNAME is present.',
            'an H&E stained image of CLASSNAME.',
            'an H&E stained image showing CLASSNAME.',
            'an H&E image showing CLASSNAME.',
            'an H&E image of CLASSNAME.',
            'CLASSNAME, H&E stain.',
            'CLASSNAME, H&E.'
            ]
# 注明生成的是图+st data
Templates = [
'CLASSNAME.',
            'a photomicrograph along with its spatial transcriptomics data showing CLASSNAME.',
            'a photomicrograph along with its spatial transcriptomics data of CLASSNAME.',
            'an image along with its spatial transcriptomics data of CLASSNAME.',
            'an image along with its spatial transcriptomics data showing CLASSNAME.',
            'an example of CLASSNAME.',
            'CLASSNAME is shown.',
            'this is CLASSNAME.',
            'there is CLASSNAME.',
            'a histopathological image along with its spatial transcriptomics data showing CLASSNAME.',
            'a histopathological image along with its spatial transcriptomics data of CLASSNAME.',
            'a histopathological photograph along with its spatial transcriptomics data of CLASSNAME.',
            'a histopathological photograph along with its spatial transcriptomics data showing CLASSNAME.',
            'shows CLASSNAME.',
            'presence of CLASSNAME.',
            'CLASSNAME is present.',
            'an H&E stained image along with its spatial transcriptomics data  of CLASSNAME.',
            'an H&E stained image along with its spatial transcriptomics data showing CLASSNAME.',
            'an H&E image along with its spatial transcriptomics data showing CLASSNAME.',
            'an H&E image along with its spatial transcriptomics data of CLASSNAME.',
            'CLASSNAME, H&E stain, spatial transcriptomics data.',
            'CLASSNAME, H&E.'
]
ffpe_templates = Templates

frozen_templates = [
    'CLASSNAME, frozen section.',
    'a photomicrograph of a frozen section showing CLASSNAME.',
    'a photomicrograph of a frozen section with its spatial transcriptomics data showing CLASSNAME.',
    'a photomicrograph of a frozen section with its spatial transcriptomics data of CLASSNAME.',
    'an image of a frozen section with its spatial transcriptomics data of CLASSNAME.',
    'an image of a frozen section with its spatial transcriptomics data showing CLASSNAME.',
    'an example of CLASSNAME in a frozen section.',
    'CLASSNAME is shown in a frozen section.',
    'this is CLASSNAME in a frozen section.',
    'there is CLASSNAME in a frozen section.',
    'a histopathological image of a frozen section with its spatial transcriptomics data showing CLASSNAME.',
    'a histopathological image of a frozen section with its spatial transcriptomics data of CLASSNAME.',
    'a histopathological photograph of a frozen section with its spatial transcriptomics data of CLASSNAME.',
    'a histopathological photograph of a frozen section with its spatial transcriptomics data showing CLASSNAME.',
    'shows CLASSNAME in a frozen section.',
    'presence of CLASSNAME in a frozen section.',
    'CLASSNAME is present in a frozen section.',
    'an H&E stained frozen section image with its spatial transcriptomics data of CLASSNAME.',
    'an H&E stained frozen section image with its spatial transcriptomics data showing CLASSNAME.',
    'an H&E image of a frozen section with its spatial transcriptomics data showing CLASSNAME.',
    'an H&E image of a frozen section with its spatial transcriptomics data of CLASSNAME.',
    'CLASSNAME, frozen section, H&E stain, spatial transcriptomics data.',
    'CLASSNAME, frozen section, H&E.'
]

st_templates = [
    'CLASSNAME, spatial transcriptomics data.',
    'CLASSNAME is shown in a spatial transcriptomics data.',
    'this is CLASSNAME in a spatial transcriptomics data.',
    'there is CLASSNAME in a spatial transcriptomics data.',
    'shows CLASSNAME in a spatial transcriptomics data.',
    'presence of CLASSNAME in a spatial transcriptomics data.',
    'CLASSNAME is present in a spatial transcriptomics data.',
]

celltype_Templates = [
    s.replace('.', ',') + 'including celltypes of CELLTYPES.'
    for s in Templates
]
