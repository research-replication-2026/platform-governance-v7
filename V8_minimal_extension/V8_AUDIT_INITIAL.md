# V8 — Audit méthodologique initial

Audit du code avant nouvelle exécution, 11 novembre 2026. Source canonique :
`experiments_v7/scripts/run_experiments_v7.py` (2 000+ lignes), tables et manifeste
dans `experiments_v7/outputs/`, manuscrit désigné par
`05_final_manuscript/CANONICAL_MANUSCRIPT_RECORD.md`.
Tous les fichiers de V7 et du manuscrit sont protégés par l'inventaire SHA-256
`V7_PRESERVATION_SHA256.json`. Aucune écriture dans ces dossiers n'est autorisée.
Le dossier `platform-governance-v7/` est vide ; ce n'est pas le simulateur actif.

## Protocole reconstruit

- Six scénarios principaux (`SCENARIOS`, lignes 170–206), deux environnements
  (non séquentiel / séquentiel complet), horizon 100 décisions.
- 180 épisodes par politique dans l'expérience principale. Budgets identiques
  en nombre de transitions, pas en coût machine. Ablations et séquentialité :
  140 épisodes ; stress : 160. Ne pas mélanger ces budgets dans A.
- Graines nominales principales d'entraînement : 1101, 1201, 1301 ; évaluation :
  5101 + 37k, k=0..9. `run_policy_set` (1168+) ajoute SEED=20261008,
  500000 pour l'environnement séquentiel et 10000 × indice du scénario.
  `train_learners` ajoute ensuite 101 / 202 / 303 / 404 par algorithme.
  Les graines d'épisodes ajoutent 13e (bandit), 17e (Q), 19e (linéaire).
  Vérifier les ensembles de graines effectives, pas seulement les listes nominales.
- Neuf comparateurs V7 : permissif, restrictif, règle de sécurité, score
  heuristique non appris, bandit contextuel, Q tabulaire, Q fin, Q linéaire,
  oracle à information privilégiée. V8 priorise bandit/Q, conserve Q linéaire
  et les quatre références déterministes ; Q fin et oracle restent diagnostiques V7.
- Les politiques utilisent toutes les masques légaux avant de choisir. Les
  interventions du garde sont donc normalement nulles ; cela ne prouve pas
  l'absence d'incidents. Le nom `mandatory_review_uncertainty` n'impose pas
  la revue : le code interdit seulement l'acceptation directe dans ce cas.

## Récompense et crédit temporel (855–989)

Le bandit estime une moyenne par état-action avec alpha=1/n, sans bootstrap,
uniquement avec `info.reward_components.current_action_reward`. Q utilise
la somme de toutes les composantes et alpha=.14, gamma=.93 ; le linéaire
alpha=.035, gamma=.93, erreur TD bornée à ±5. Exploration commune
epsilon=.04+.24 exp(-e/(episodes/3)). Le code confirme ces propriétés.

La récompense totale additionne gains de qualité des contributions légitimes,
coûts des actions, exposition nocive, blocage nocif, restrictions de contributions
légitimes, incidents, appels et liquidation. `collective_state_reward` vaut
toujours zéro ; confiance et participation changent les soumissions futures.
L'objectif commun d'évaluation doit rester la récompense totale / horizon.

**Limite du comparateur** : en environnement séquentiel, un contrôle/revue donne
au bandit son coût immédiat sans le résultat différé de cette même soumission.
Ce n'est pas un bandit à récompense retardée réattribuée à l'action d'origine.
La comparaison porte sur ce bandit myope précis, pas sur toute la famille.

## Informations disponibles (398–433, 806–843)

Vecteur de 17 composantes : risque, incertitude, participation, acceptations
légitimes récentes, charge de revue, capacité restante, exposition, incidents
cumulés, appels en attente, restrictions injustifiées cumulées, confiance,
charge automatique, risque moyen des revues, délai moyen des revues,
charge des incidents en attente, sévérité moyenne de ces incidents, charge
des appels. Plusieurs composantes sont des agrégats privilégiés du simulateur
(incidents à venir déjà en file, exposition vraie, restrictions réellement
injustifiées), pas des mesures empiriquement validées disponibles en production.
V8 conserve cet héritage en C1 et le signale explicitement.

**Écart important entre observation brute et entrée effective** : la
discrétisation tabulaire ne retient que les indices
0,1,2,4,5,6,7,8,10,11,14. Les indices 3,9,12,13,15,16 sont ignorés par
le bandit et Q tabulaires, mais utilisés par le linéaire. A ne peut donc pas
être décrit comme une comparaison à représentation effective identique entre
tabulaire et linéaire. Le bandit et Q partagent bien la même représentation.
Nocivité, qualité réelle, identité de groupe et détails individuels des files
ne sont pas directement transmis aux politiques déployables ; l'oracle voit
la nocivité et est exclu de V8.

## Dynamique et mécanismes (296–805)

Une soumission par décision, six actions. Files de contrôles automatiques
(délai 1), revues humaines (2), incidents (2–6), appels (4). Seuls les objets
déjà présents au début de la transition avancent. La confiance répond aux
acceptations légitimes, incidents et restrictions injustifiées ; la participation
suit une cible de confiance/pressions puis change la distribution de nocivité
et de diversité. Elle ne modifie PAS le nombre de soumissions (toujours 100).
`unresolved_incidents` est un compteur cumulatif sans mécanisme de résolution.
Il ne doit pas être interprété comme un véritable stock d'incidents ouverts.
La capacité de revue peut être peu contraignante à une arrivée par décision ;
les scénarios principaux ne garantissent pas la saturation. Les files et les
capacités réelles sont communes à toutes les politiques d'un scénario.

## Deux défauts à corriger uniquement dans V8

1. **Bootstrap terminal indu** : Q et Q linéaire utilisent gamma × valeur de
   l'observation finale même lorsque `truncated=True`, alors que la liquidation
   rend l'horizon terminal. Le résidu n'est pas nul. V8 annule ce bootstrap.
2. **Liquidation incomplète** : `_liquidate_terminal` vide les incidents AVANT
   les files de revue/contrôle. Résoudre une soumission nocive peut créer un
   nouvel incident après ce passage. Ce nouvel incident reste non payé à la fin.
   V8 matérialise aussi ces incidents créés pendant la liquidation, sans ajouter
   de nouvelles décisions ni modifier les autres mécanismes.

Ces corrections empêchent de traiter silencieusement les lignes V7 comme des
lignes corrigées V8. Prévoir un bras A de réplication exacte V7 et un bras A
corrigé séparé ; B/C utilisent exclusivement la sémantique corrigée. La différence
entre ces deux bras est une sensibilité d'implémentation, pas un effet de feedback.

## Appariement et inférence

Les mêmes graines d'évaluation sont utilisées entre politiques, mais un seul RNG
sert aux arrivées et aux événements conditionnels aux actions. Des actions
différentes consomment des tirages différents : même graine ne signifie PAS
même suite exogène de soumissions. Conserver cette limite pour reproduire V7 ;
ne pas prétendre disposer d'un appariement parfait des chocs exogènes. L'état
endogène fait en outre varier les distributions de soumissions.

Les tableaux V7 `paired_differences` (1270+) utilisent 1.96 × SD/sqrt(30),
avec 3 entraînements × 10 évaluations. Cela sous-estime potentiellement
l'incertitude entre entraînements. La validation statistique ultérieure dans
`05_final_manuscript/FINAL_STATISTICAL_VALIDATION_V7.md` corrige ce point avec
trois moyennes d'entraînement et t(df=2), mais le manuscrit canonique garde
les anciens intervalles comme résultats principaux. Il indique 8 intervalles
négatifs et 4 indéterminés dans sa sensibilité, alors que la table de validation
doit être recomptée à partir du CSV, sans reprendre ce décompte de mémoire.

Les références déterministes sont réévaluées à l'identique sous chaque étiquette
d'entraînement : ces duplications ne créent aucune réplication indépendante.
Les intervalles V8 seront conditionnels aux dix scénarios aléatoires d'évaluation
fixés ; ils quantifieront la variabilité des entraînements, sans transformer
300 évaluations en 300 entraînements. Les tests d'interaction B/C seront des
différences de différences au niveau des graines, avec contrôle de multiplicité.

## État avant exécution

Audit du code effectué. Exécutions, contrôles des graines effectives, preuve
numérique des défauts, reproduction ciblée et estimation de charge : à réaliser
dans V8 et à consigner dans `V8_EXECUTION_REPORT.md` avant les campagnes longues.
Pas de reconstruction du corpus documentaire, pas de modification du manuscrit
existant. Aucune conclusion nouvelle n'est encore établie.


## Inventaire compl?mentaire des ?l?ments V7 conserv?s

- `experiments_v7/README.md` fournit les deux commandes de reproduction
  (tests unitaires puis script principal). Le script principal ?crit dans V7 :
  ne le relancer que dans une copie isol?e, conform?ment ? la pr?servation.
- `experiments_v7/tests/test_v7_environment.py` contient cinq tests couvrant
  attribution de r?compense, cible du bandit, liquidation, forme d'observation
  et diagnostic d'?tats jamais vus. Ils ont ?t? ex?cut?s sans modification.
- Les configurations de simulation sont des dataclasses et constantes dans le
  script V7 ; les param?tres principaux ne sont pas lus dans le fichier racine
  `seed_config.json`. Le manifeste V8 s?rialise ces configurations effectives.
- Les fichiers `v7_main_*`, `v7_paired_differences`, `v7_reward_attribution_*`,
  `v7_training_*`, `v7_ablation_*`, `v7_sequentiality_*`, `v7_stress_sensitivity_*`,
  `v7_security_comparable_*` et les audits de validit? restent disponibles dans
  `experiments_v7/outputs/tables/`. Les rapports et quatre figures V7 restent
  inchang?s dans les sous-dossiers correspondants.
- `run_q_diagnostics` explore les budgets 40/120/360 et trois jeux de
  alpha/gamma/epsilon, sur trois sc?narios et deux graines (54 ajustements).
  Sa formule de graine d?pend du budget : il ne s'agit pas de prolonger le m?me
  apprentissage sur un pr?fixe al?atoire commun. Cette sensibilit? ne suffit
  donc pas ? isoler parfaitement un effet causal du budget. V8 n'utilise pas
  ces lignes comme r?p?titions principales et fixe le budget ? 180 partout.
- `experiments_v7/automated_validation/run_automated_validation.py` concerne
  le corpus documentaire, pas la comparaison de politiques. Ses sorties
  identifient des m?canismes dans 32 ?v?nements et des diagnostics de coh?rence
  automatis?e ; elles ne calibrent ni les transitions ni les probabilit?s du
  simulateur et ne constituent pas un double codage humain. Aucune de ces
  donn?es n'a ?t? reconstruite ni modifi?e dans V8.


## Verification addendum after execution

The original three-training-seed grouped interval count was independently
confirmed: eight negative, zero positive, four zero-crossing cells. All 510
targeted saved-output comparisons matched to 2.85e-14. Effective-seed audits
found no within-policy cross-training episode collisions and no train/evaluation
overlap. A later prospective amendment additionally removed historical cross-algorithm/cross-group seed coupling from primary inference: labels 10001..10030 now form disjoint paired blocks, and the three historical labels remain diagnostic. The complete V8 campaign is now verified; all 138
protected files retain their initial hashes and no additional files were created
in their directories. See the execution report for completion counts and the
results summary for statistical findings. The two V8 corrections remain
separate from the unchanged exact V7 stratum.
