# Discours commercial — Présentation ALPR
## Durée totale : 8 minutes | 4 intervenants × 2 minutes

---

## INTERVENANT 1 — Slides 1, 2 et 3 (2 minutes)

### Slide 1 — Titre (5 secondes)

*[Slide d'accroche, pas de discours — transition directe]*

---

### Slide 2 — Le problème au quotidien (~55 secondes)

Bonjour à tous et merci de nous accorder ces quelques minutes.

Imaginez la scène : vous arrivez à votre hôtel après une longue route, il pleut, il fait nuit. Vous êtes devant la barrière du parking. Et là, il faut sortir de la voiture, chercher votre code, taper un digicode sous la pluie... Quand ça fonctionne du premier coup, c'est déjà pénible. Quand ce n'est pas le cas, c'est franchement frustrant.

Ce scénario, ce n'est pas un cas isolé. C'est le quotidien de milliers de clients d'hôtels, de campings, de résidences privées. Chaque véhicule qui s'arrête, c'est une file d'attente qui s'allonge, une expérience client qui se dégrade avant même d'avoir commencé.

Et si on pouvait supprimer cette friction ? Et si l'accès au parking devenait totalement automatique, fluide, invisible ?

---

### Slide 3 — Notre réponse : ALPR (~60 secondes)

C'est exactement ce que nous avons construit. Notre solution ALPR — pour Automatic License Plate Recognition — automatise intégralement le contrôle d'accès.

Le principe est simple : le véhicule approche, la caméra détecte la plaque, notre intelligence artificielle l'analyse en temps réel, et si le numéro est autorisé, la barrière s'ouvre. Sans intervention humaine. Sans code. Sans attente.

Trois points essentiels à retenir sur notre solution. Premièrement, elle est 100% autonome : une fois configurée, elle gère les accès toute seule. Deuxièmement, elle est conçue dans le strict respect du RGPD et du cadre réglementaire français. Et troisièmement, elle intègre un double niveau de vérification — plaque et véhicule — pour une sécurité renforcée.

Je laisse maintenant la parole à mon collègue pour le volet réglementaire et la présentation de l'application.

---

## INTERVENANT 2 — Slides 4 et 5 (2 minutes)

### Slide 4 — Cadre réglementaire (~50 secondes)

Merci. Avant de vous présenter l'application en détail, un point essentiel : la conformité légale.

Les systèmes de lecture de plaques sont encadrés par une réglementation très stricte, et c'est normal. Nous avons étudié en profondeur le cadre juridique, tant français qu'européen, pour construire une solution parfaitement conforme.

Côté France, le Code de la Sécurité Intérieure encadre les usages LAPI. La CNIL impose des durées de conservation strictes. Pour un usage privé comme la gestion de parking, la règle est claire : effacement immédiat des données dès la fin de la transaction.

Côté Europe, notre solution respecte les principes du RGPD — minimisation des données, transparence, limitation de conservation. Et nous sommes alignés avec l'AI Act, le nouveau règlement européen sur l'intelligence artificielle, qui classe ces systèmes en haut risque et impose documentation technique et traçabilité.

Un document réglementaire complet est disponible en annexe pour ceux qui souhaitent approfondir.

---

### Slide 5 — L'application ALPR (~70 secondes)

Concrètement, comment ça fonctionne pour vous en tant qu'opérateur ?

Notre application repose sur une API client simple. Vous y enregistrez les numéros de plaque de vos clients autorisés. À partir de là, le système est totalement autonome.

L'environnement de lecture est extrêmement stable — une caméra fixe, un angle contrôlé, des conditions maîtrisées. Résultat : un taux de reconnaissance de 97% toutes conditions confondues. Les seuls cas d'échec sont les plaques physiquement détériorées ou masquées.

Et pour ces 3% restants, nous avons prévu un digicode de secours. Aucun client ne reste bloqué.

L'API est conçue pour s'intégrer facilement à vos systèmes existants — PMS hôtelier, logiciel de réservation, gestion de camping. C'est une brique qui vient se greffer à votre infrastructure sans la remplacer.

Je passe la parole pour la partie technique.

---

## INTERVENANT 3 — Slides 6, 7 et 8 (2 minutes)

### Slide 6 — Architecture du modèle (~40 secondes)

Merci. Entrons dans le moteur de notre solution.

Notre pipeline repose sur deux branches indépendantes. La branche principale, celle qui gère l'accès, enchaîne deux modèles : un YOLOv8 nano fine-tuné qui détecte et isole la plaque dans l'image, puis un modèle Fast Plate OCR spécialisé qui lit le numéro.

La branche secondaire détecte le véhicule lui-même et identifie sa marque grâce à un modèle EfficientNet B4 entraîné sur un dataset de véhicules européens.

Tous nos modèles sont fine-tunés spécifiquement pour le marché européen.

---

### Slide 7 — Détection & Lecture (~35 secondes)

Un zoom rapide sur les deux composants clés. YOLOv8 nano est un modèle léger, conçu pour le temps réel. Il découpe avec précision la zone de la plaque, même dans des conditions de luminosité variées.

Fast Plate OCR prend le relais pour la lecture du numéro. C'est un OCR spécialisé, pas un OCR générique. Il a été fine-tuné sur des formats de plaques européennes pour une fiabilité maximale. En cas d'échec, le système bascule automatiquement sur le digicode de secours et log l'image pour levée de doute.

---

### Slide 8 — Sécurité & Cas limites (~45 secondes)

Un mot sur la sécurité, car nous avons identifié un risque : la présentation d'une plaque fictive portant un numéro autorisé. C'est pour contrer cette menace que nous avons développé la branche de détection véhicule et marque. Elle compare automatiquement le véhicule détecté à l'information déclarée par le client et signale toute anomalie à l'opérateur.

Notre système fonctionne en trois niveaux de sécurité : d'abord la reconnaissance de plaque à 97%, puis la vérification croisée marque et véhicule, et enfin le digicode de secours avec log systématique.

Chaque niveau renforce le précédent. On passe la main pour les perspectives d'évolution.

---

## INTERVENANT 4 — Slides 9 et 10 (2 minutes)

### Slide 9 — Évolutions futures (~80 secondes)

Merci. Ce que vous avez vu aujourd'hui, c'est notre version 1 — un système fonctionnel, fiable, conforme.

Mais notre vision va plus loin. En version 2, la prochaine étape, nous allons rendre la classification véhicule pleinement discriminante. Aujourd'hui, elle signale les anomalies à l'opérateur. Demain, elle fera partie intégrante de la chaîne d'autorisation : si la marque du véhicule ne correspond pas à la fiche client, l'accès est automatiquement refusé. C'est un jalon de sécurité majeur.

En version 3, notre vision à moyen terme, nous prévoyons un dashboard opérateur en temps réel avec des analytics d'usage — nombre d'accès, taux de reconnaissance, alertes — et surtout une capacité d'intégration multi-sites pour les groupes hôteliers ou les chaînes de campings qui souhaitent centraliser la gestion de leurs accès.

---

### Slide 10 — Merci (~40 secondes)

Pour conclure, notre solution ALPR répond à un besoin concret : fluidifier l'accès aux parkings privés tout en garantissant sécurité et conformité réglementaire.

97% de fiabilité. 100% autonome. Conforme RGPD et AI Act.

Merci pour votre temps et pour l'attention que vous avez portée à notre projet. Nous sommes à votre entière disposition pour répondre à vos questions et approfondir les points qui vous intéressent.

Le document réglementaire LAPI complet est disponible en annexe de cette présentation.

---

*Durée estimée par intervenant :*
- **Intervenant 1** (Slides 1-3) : ~2 min
- **Intervenant 2** (Slides 4-5) : ~2 min
- **Intervenant 3** (Slides 6-8) : ~2 min
- **Intervenant 4** (Slides 9-10) : ~2 min
