import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]

def parents(people, person, one_gene, two_genes):
    """
    return the probabbilities of the mother and father giving the child a gene
    """

    prob = dict()
    if people[person]["mother"] in one_gene:
        prob["mother"] = 0.5
    elif people[person]["mother"] in two_genes:
        prob["mother"] = 0.99
    else:
        prob["mother"] = 0.01

    if people[person]["father"] in one_gene:
        prob["father"] = 0.5
    elif people[person]["father"] in two_genes:
        prob["father"] = 0.99
    else:
        prob["father"] = 0.01

    return prob

def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    joint = 1
    for person in people:
        if person in one_gene:
            if people[person]["mother"] == None:
                prob = PROBS["gene"][1]
            else:
                temp = parents(people, person, one_gene, two_genes)
                prob = (temp["mother"] * (1 - temp["father"])) + (temp["father"] * (1 - temp["mother"]))
                #print(prob, "before trait")
            gene = 1

        elif person in two_genes:
            if people[person]["mother"] == None:
                prob = PROBS["gene"][2]
            else:
                temp = parents(people, person, one_gene, two_genes)
                prob = temp["mother"] * temp["father"]
            gene = 2

        else:
            if people[person]["mother"] == None:
                prob = PROBS["gene"][0]
            else:
                temp = parents(people, person, one_gene, two_genes)
                prob = (1 - temp["mother"]) * (1 - temp["father"])
            gene = 0

        if person in have_trait:
            prob = prob * PROBS["trait"][gene][True]
        else:
            prob = prob * PROBS["trait"][gene][False]

        joint *= prob
        #print(prob, person, PROBS["trait"][gene])

        #print(one_gene, two_genes, have_trait)
        #print(joint)

    return joint


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p

        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p
            

def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    totalgene = 0
    for person in probabilities:
        totalgene = 0
        totaltrait = 0
        for temp in probabilities[person]["gene"]:
            totalgene += probabilities[person]["gene"][temp]
        #print(probabilities[person]["gene"])
        for temp in probabilities[person]["gene"]:
            probabilities[person]["gene"][temp] /= totalgene
        #print(probabilities[person]["gene"])
        totaltrait = probabilities[person]["trait"][True] + probabilities[person]["trait"][False]
        probabilities[person]["trait"][True] /= totaltrait
        probabilities[person]["trait"][False] /= totaltrait 


if __name__ == "__main__":
    main()
