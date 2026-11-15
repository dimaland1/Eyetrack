"""Numéro de version d'Eyetrack (versionnage sémantique : majeur.mineur.correctif).

Une version publiée porte l'étiquette git « v<VERSION> ». tools/build_exe.py n'écrit ce numéro
tel quel que si le commit construit porte cette étiquette et que rien n'est modifié ; sinon la
construction est marquée « <VERSION>-dev+<commit> », pour ne jamais la confondre avec la version
publiée.
"""

VERSION = "0.1.0"
