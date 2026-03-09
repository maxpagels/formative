Motivation
==========

Causal estimation is an incredibly fascinating topic. What causes what is one of the great questions,
with implications for everything from science to policy to personal decision-making. You'll find plenty
of resources on the topic, but there are scarce resources on how causal estimation is necessarily a blend
of art, science and problem-framing.

The first thing to understand about causal estimation is that all causal problems are based on *assumptions*.
YouTube has a `classic clip by Richard Feynman <https://www.youtube.com/watch?v=Dp4dpeJVDxs>`_ answering a
question about why magnets attract and repel. The interviewer grows impatient with Feynman's answer, which is
that you can feel magnets pushing apart from each other because they repel each other. This short answer doesn't
satisfy the interviewer, who wants a deeper explanation, which leads Feynman to explain the dangerousness of
asking "why?" – it leads to an infinite chain of asking why. If you want to understand why a customer bought a
product, it may be because they wanted it. And they may have wanted it because they like the colour. The colour, in
turn, may be likeable because it reminds them of a fond memory. And why they have that fond memory is because
their brain recalled it. Which is because of the way their brain works. And so on. At some point, you will start
questioning the universe itself, unless you simply accept that some things are just assumed to be true. This also
implies that some assumptions are not *testable*; it is simply not possible to gather data to confirm or refute them.

Causal assumptions differ based on the problem you are trying to solve, the method you have chosen, and the data
you have available. There are always assumptions that you yourself make and argue for, and there are assumptions
that you inherit from the method you have chosen. A classic example is the so-called SUTVA assumption. If you show
variant A to a customer, and variant B to another customer, customer B may change their behaviour because they know
that variant A exists (maybe their friend told them about it). This messes up your estimation, and is also something
you cannot reasonably test using collected data. You assume it holds, based on experience and knowledge. Someone else
might assume it doesn't hold, and that is a perfectly valid assumption to make. Art is subjective, and so is the art
side of causal estimation.

The second part of causal estimation that I find exciting is the methods you use and the ingenious ways they frame
and exploit problem settings in order to control for other factors influencing the outcome. These can be coarsely
divided into three categories: *pseudo-randomisation*, quasi-randomisation, and what I would call "intelligent design".

Pseudo-randomisation is the gold standard, and is the basis of the classic A/B test — if we randomly assign customers
to variant A or variant B, then we can claim the difference in outcomes cannot be explained by any other factor than
treatment, because it is exceedingly unlikely that other factors would systematically differ between groups. We use the
term "pseudo"-randomisation here because no randomisation algorithm made on a computer is ever truly random.

Quasi-randomisation is a clever way to find structure that *mimics* randomisation. Let's say you launch a new version
of your website, and this new version is rolled out to all customers at once because the complexity of rolling it out
via A/B testing is too high. If you know the exact time of the rollout, and didn't tell anyone it was going to happen,
you can bucket the users online just before and just after the rollout. There is a good chance there is no systematic
difference between these two groups of users – from their perspective, it is down to luck which version they saw. The
only difference is the treatment, and the rest is "quasi-random", allowing us to make claims about causality on similar
grounds to pseudo-randomisation.

Intelligent design is a more generic term, because methods that fall into this category use a variety of interesting
techniques. Notably these do *not* rely on making a "random-like" assumption. One method to highlight because of its
beautiful design is instrumental variables. If we want to know if education causes income, we can control for confounding
factors by assuming that some other phenomenon causes (at least partially) education but does not directly cause income.
And we do mean *causes*, not just correlates. If we claim the distance to the nearest college will (at least partially)
have a causal effect on education attendance, and further claim that a mere distance to college will not make you earn
more, we have found a valid instrument that we can elegantly use to eliminate confounding.

I hope this introduction to causal estimation has given you a sense of the blend of art, science and problem-framing
that it requires. I find beauty in the ways we exploit the structure of problems to understand more about the world,
and I hope you do too.
