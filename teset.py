from nacl.bindings.utils import sodium_memcmp
import nacl.encoding
import nacl.hash
import random


def getOperations(workLoad):
    cWorkLoad = workLoad
    if 'pseudorandom' in cWorkLoad:
        seedAndNoOfRequests = cWorkLoad[:-1].split('(')[1].split(',')
        return generatePseudoRandomRequests(int(seedAndNoOfRequests[0]), int(seedAndNoOfRequests[1]))
    else:
        return cWorkLoad.split(';')

def generatePseudoRandomRequests(rSeed, noOfRequests):
    listofRequest = ["put('movie','star')", "append('movie',' wars')", "get('movie')", "put('jedi,'luke skywalker)",
                     "slice('jedi','0:4')", "get('jedi')"]
    random.seed(rSeed)
    requests = random.sample(listofRequest, k=noOfRequests)
    return requests

operations = getOperations('pseudorandom(233,5)')
operations1 = getOperations("put('movie','star'); append('movie',' wars'); get('movie')")
print(operations)
print(operations1)
