"""Hand party labels for BASIL span targets (auditable). R = Republican/conservative figure or body,
D = Democratic/liberal figure or body, N = non-partisan / institution / other."""
R = """Donald_Trump Paul_Ryan Chris_Christie Republican_Lawmakers Peter_Strzok:N Asa_Hutchinson Michael_Steele Marco_Rubio
Paul_LePage Republican_lawmakers Trump Roy_Moore Roger_Stone Raul_Labrador Glenn_Beck Republicans Rick_Perry Michael_Grimm
Mitt_Romney Senate_Republicans Todd_Akin Rick_Santorum Mike_Huckabee Elizabeth_Lauten Michael_Flynn Mike_Pence GOP_committee_members
Tea_Party James_Mattis Joe_Arpaio David_Brat Blake_Farenthold Duncan_and_Margaret_Hunter Romney_campaign Secure_America_Now
Donald_Trump_Jr. House_Republicans Eric_Cantor Ted_Cruz Mary_Cheney William_Barr Neil_Gorsuch Alabama_Republicans Liz_Cheney
Chuck_Grassley fiscal_conservatives George_W._Bush Jan_Brewer Kevil_McCarthy Stephen_Fincher John_Boehner Steve_Scalise
Michele_Bachmann Jeb_Bush Duncan_Hunter Michael_Cohen National_Rifle_Association Seth_Hutchinson"""
D = """Barack_Obama Hillary_Clinton Joe_Biden Democratic_Lawmakers Ruth_Bader_Ginsburg House_Democrats Nancy_Pelosi Rashida_Tlaib
Obama_administration Bob_Etheridge Obama_campaign Adam_Schiff Anthony_Weiner Sasha_and_Malia_Obama Kathy_Griffin Richard_Blumenthal
Democratic_lawmakers William_Cowan Bernie_Sanders Democrats Lois_Lerner Democrats_presidential_candidates Bruce_Braley
Obama's_administration Larry_Summers Jesse_Jackson Michael_Bloomberg Alexandria_Ocasio-Cortez Eric_Holder Susan_Rice
Senate_Democrats Harry_Reid senate_Democrats Chuck_Schumer Lois_lerner Deval_Patrick Victoria_Nuland Joe_Lieberman"""
PARTY = {}
for blk, lab in ((R, "R"), (D, "D")):
    for tok in blk.split():
        name, _, override = tok.partition(":")
        PARTY[name] = override or lab
party = lambda t: PARTY.get(t, "N")
