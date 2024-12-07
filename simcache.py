#!/usr/bin/python3

"""
Arnik Shah
CS-UY 2214
December 1, 2024
simcache.py
"""

from collections import namedtuple
import re
import argparse

#####################################
# Helper Functions for E20 Simulator#
#####################################

# Some helpful constant values that we'll be using.
Constants = namedtuple("Constants",["NUM_REGS", "MEM_SIZE", "REG_SIZE"])
constants = Constants(NUM_REGS = 8,
                      MEM_SIZE = 2**13,
                      REG_SIZE = 2**16)

def load_machine_code(machine_code, mem):
    """
    Loads an E20 machine code file into the list
    provided by mem. We assume that mem is
    large enough to hold the values in the machine
    code file.
    sig: list(str) -> list(int) -> NoneType
    """
    machine_code_re = re.compile("^ram\[(\d+)\] = 16'b(\d+);.*$")
    expectedaddr = 0
    for line in machine_code:
        match = machine_code_re.match(line)
        if not match:
            raise ValueError("Can't parse line: %s" % line)
        addr, instr = match.groups()
        addr = int(addr,10)
        instr = int(instr,2)
        if addr != expectedaddr:
            raise ValueError("Memory addresses encountered out of sequence: %s" % addr)
        if addr >= len(mem):
            raise ValueError("Program too big for memory")
        expectedaddr += 1
        mem[addr] = instr

def print_state(pc, regs, memory, memquantity):
    """
    Prints the current state of the simulator, including
    the current program counter, the current register values,
    and the first memquantity elements of memory.
    sig: int -> list(int) -> list(int) - int -> NoneType
    """
    print("Final state:")
    print("\tpc="+format(pc,"5d"))
    for reg, regval in enumerate(regs):
        print(("\t$%s=" % reg)+format(regval,"5d"))
    line = ""
    for count in range(memquantity):
        line += format(memory[count], "04x")+ " "
        if count % 8 == 7:
            print(line)
            line = ""
    if line != "":
        print(line)

def getOpCode(instr):
    """
    Extract Opcode
    sig: int -> int
    """
    code = instr >> 13
    return code & 0b111

def getLastFourBits(instr):
    """
    Get last four bits
    sig: int -> int
    """
    return instr & 0b1111

def getLastSevenBits(instr):
    """
    Get last seven bits signed
    sig: int -> int
    """
    num = instr & 0b111111
    isNeg = instr & 0b1000000
    if isNeg:
        num -= 2**6
    return num

def getLastSevenBitsUnsigned(instr):
    """
    Get last seven bits unsigned
    sig: int -> int
    """
    return instr & 0b1111111

def getLastThirteenBits(instr):
    """
    Get last thirteen bits
    sig: int -> int
    """
    return instr & 0b1111111111111

def incPC(pc):
    """
    increment pc
    sig: int -> int
    """
    return fixPC(pc + 1)

def fixPC(pc):
    """
    get pc in range
    sig: int -> int
    """
    if pc >= constants.REG_SIZE:
        pc -= constants.REG_SIZE
    elif pc < 0:
        pc = constants.REG_SIZE + pc
    return pc

def getRegALocation(instr):
    """
    Get regA location
    sig: int -> int
    """
    code = instr >> 10
    return code & 0b111

def getRegBLocation(instr):
    """
    Get regB location
    sig: int -> int
    """
    code = instr >> 7
    return code & 0b111

def getRegCLocation(instr):
    """
    Get regC location
    sig: int -> int
    """
    code = instr >> 4
    return code & 0b111

def makeUnsigned(val):
    """
    make signed num
    sig: int -> int
    """
    if val >= constants.REG_SIZE:
        val = val - constants.REG_SIZE
    elif val < 0:
        val = constants.REG_SIZE + val
    return val

def signExtend(val):
    """
    sign extend num
    sig: int -> int
    """
    if val & 0b1000000:
        return val | 0b1111111110000000
    return val

#####################################
# Helper Functions for E20 Cache#####
#####################################

def print_cache_config(cache_name, size, assoc, blocksize, num_rows):
    """
    Prints out the correctly-formatted configuration of a cache.

    cache_name -- The name of the cache. "L1" or "L2"

    size -- The total size of the cache, measured in memory cells.
        Excludes metadata

    assoc -- The associativity of the cache. One of [1,2,4,8,16]

    blocksize -- The blocksize of the cache. One of [1,2,4,8,16,32,64])

    num_rows -- The number of rows in the given cache.

    sig: str, int, int, int, int -> NoneType
    """

    summary = "Cache %s has size %s, associativity %s, " \
        "blocksize %s, rows %s" % (cache_name,
        size, assoc, blocksize, num_rows)
    print(summary)

def print_log_entry(cache_name, status, pc, addr, row):
    """
    Prints out a correctly-formatted log entry.

    cache_name -- The name of the cache where the event
        occurred. "L1" or "L2"

    status -- The kind of cache event. "SW", "HIT", or
        "MISS"

    pc -- The program counter of the memory
        access instruction

    addr -- The memory address being accessed.

    row -- The cache row or set number where the data
        is stored.

    sig: str, str, int, int, int -> NoneType
    """
    log_entry = "{event:8s} pc:{pc:5d}\taddr:{addr:5d}\t" \
        "row:{row:4d}".format(row=row, pc=pc, addr=addr,
            event = cache_name + " " + status)
    print(log_entry)

class LRUCache:
    # Nested class used to implement Doubly Linked List with cache
    class Node:
        def __init__(self, vals, next = None, prev = None, key = None):
            """
            intitalize node class the stores vals, next ptr, prev ptr, and key in hash table
            sig: list[int], Node, Node, (int, int) -> None
            """
            self.vals = vals
            self.next = next
            self.prev = prev
            self.key = key

    def __init__(self, assoc, blocksize):
        """
        intitalize LRU Cache with has table the maps ValidBit and Tag as keys and Nodes as values
        node are organized in a doubly linked list with dummy nodes
        sig: int, int -> None
        """
        self.capacity = assoc
        self.currSize = 0
        self.blocksize = blocksize
        # dictionary that has ValidBit and Tag as keys and Nodes as values
        self.data = {}
        # Dummy nodes for head and tail
        self.tail = self.Node(0)
        self.head = self.Node(0, self.tail, None)
        self.tail.prev = self.head

    def get(self, addr, rows):
        """
        Get value from cache 
        sig: int, int -> int, None
        """
        if self.currSize != 0:
            tag = (addr // self.blocksize) // rows
            if (1, tag) in self.data.keys():
                # Update Doubly Linked List
                node = self.data[(1, tag)]
                prevNode = node.prev
                nextNode = node.next
                prevNode.next = nextNode
                nextNode.prev = prevNode
                self.addHead(node)
                # Access item with offset
                offset = addr % self.blocksize
                return self.data[(1, tag)].vals[offset]
            return None
        return None

    def put(self, addr, memory, rows):
        """
        Put value from cache 
        sig: int, list[int], int -> None
        """
        # If cache is full, evict the LRU
        if self.currSize == self.capacity:
            self.evict()
        # data at (ValidBit, Tag) gives block
        tag = (addr // self.blocksize) // rows
        start = (addr // self.blocksize) * self.blocksize
        end = ((addr // self.blocksize) * self.blocksize) + self.blocksize
        # Update Dictionary
        self.data[(1, tag)] = self.Node(memory[start : end], None, None, (1, tag))
        # Update Doubly Linked List
        self.addHead(self.data[(1, tag)])
        self.currSize += 1

    def evict(self):
        """
        Evict LRU value from cache 
        sig: None -> None
        """
        # Remove data from Doubly Linked List 
        LRUNode = self.tail.prev
        LRUNode.prev.next = self.tail
        self.tail.prev = LRUNode.prev
        # Remove data from dictionary
        del self.data[LRUNode.key]
        self.currSize -= 1

    def addHead(self, node):
        """
        Make node the head in the doubly linked list
        sig: None -> None
        """
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

def lw(memory, regs, addr):
    """
    Orginal LW implmentation with no cache
    sig: list[int], list[int], int -> None
    """
    regDst = getRegBLocation(addr)
    regSrc = getRegALocation(addr)
    imm = getLastSevenBits(addr)
    if regDst == 0:
        return
    loc = regs[regSrc] + imm
    loc = getLastThirteenBits(loc)
    regs[regDst] = makeUnsigned(memory[loc])
    return

# LW with L1 cahce
def lwL1(memory, regs, addr, L1Cache, L1blocksize, L1rows, pc):
    """
    LW implmentation with L1 cache
    sig: list[int], list[int], int, int, int, int, int -> None
    """
    regDst = getRegBLocation(addr)
    regSrc = getRegALocation(addr)
    imm = getLastSevenBits(addr)
    loc = regs[regSrc] + imm
    loc = getLastThirteenBits(loc)
    # Check Cache
    row = (loc // L1blocksize) % L1rows
    value = L1Cache[row].get(loc, L1rows)
    if value is not None:
        # Cache Hit
        print_log_entry("L1", "HIT", pc, loc, row)
    else:
        # Cache Miss
        print_log_entry("L1", "MISS", pc, loc, row)
        value = memory[loc]
        L1Cache[row].put(loc, memory, L1rows)
    # Do not modify register 0
    if regDst == 0:
        return
    regs[regDst] = makeUnsigned(value)
    return

# LW with L1 and L2 cache
def lwL1L2(memory, regs, addr, L1Cache, L1blocksize, L1rows, L2Cache, L2blocksize, L2rows, pc):
    """
    LW implmentation with L1 and L2 cache
    sig: list[int], list[int], int, int, int, int, int, int, int, int -> None
    """
    regDst = getRegBLocation(addr)
    regSrc = getRegALocation(addr)
    imm = getLastSevenBits(addr)
    loc = regs[regSrc] + imm
    loc = getLastThirteenBits(loc)
    # Check L1 Cache
    row = (loc // L1blocksize) % L1rows
    value = L1Cache[row].get(loc, L1rows)
    if value is not None:
        # L1 Cache Hit
        print_log_entry("L1", "HIT", pc, loc, row)
    else:
        # L1 Cache Miss
        print_log_entry("L1", "MISS", pc, loc, row)
        # Check L2 Cache
        rowL2 = (loc // L2blocksize) % L2rows
        value = L2Cache[rowL2].get(loc, L2rows)
        if value is not None:
            # L2 Cache Hit
            print_log_entry("L2", "HIT", pc, loc, rowL2)
            # Store Data in L1 Cache
            L1Cache[row].put(loc, memory, L1rows)
        else:
            # L2 Cache Miss
            print_log_entry("L2", "MISS", pc, loc, rowL2)
            # Store Data in L2 and L1 Cache
            value = memory[loc]
            L2Cache[rowL2].put(loc, memory, L2rows)
            L1Cache[row].put(loc, memory, L1rows)
    # Do not modify register 0
    if regDst == 0:
        return
    regs[regDst] = makeUnsigned(value)
    return

def sw(memory, regs, addr):
    """
    Orginal SW implmentation with no cache
    sig: list[int], list[int], int -> None
    """
    regSrc = getRegBLocation(addr)
    regAddr = getRegALocation(addr)
    imm = getLastSevenBits(addr)
    loc = regs[regAddr] + imm
    loc = getLastThirteenBits(loc)
    memory[loc] = regs[regSrc]
    return

# SW with L1 cache
def swL1(memory, regs, addr, L1Cache, L1blocksize, L1rows, pc):
    """
    SW implmentation with L1 cache
    sig: list[int], list[int], int, int, int, int, int -> None
    """
    regSrc = getRegBLocation(addr)
    regAddr = getRegALocation(addr)
    imm = getLastSevenBits(addr)
    loc = regs[regAddr] + imm
    loc = getLastThirteenBits(loc)
    # Modify value in memory
    memory[loc] = regs[regSrc]
    # Add value to cache
    row = (loc // L1blocksize) % L1rows
    L1Cache[row].put(loc, memory, L1rows)
    print_log_entry("L1", "SW", pc, loc, row)
    return

# SW with L1 and L2 cache
def swL1L2(memory, regs, addr, L1Cache, L1blocksize, L1rows, L2Cache, L2blocksize, L2rows, pc):
    """
    SW implmentation with L1 and L2 cache
    sig: list[int], list[int], int, int, int, int, int, int, int, int-> None
    """
    regSrc = getRegBLocation(addr)
    regAddr = getRegALocation(addr)
    imm = getLastSevenBits(addr)
    loc = regs[regAddr] + imm
    loc = getLastThirteenBits(loc)
    # Modify value in memory
    memory[loc] = regs[regSrc]
    # Add value to L1 cache
    row = (loc // L1blocksize) % L1rows
    L1Cache[row].put(loc, memory, L1rows)
    print_log_entry("L1", "SW", pc, loc, row)
    # Add value to L2 cache
    rowL2 = (loc // L2blocksize) % L2rows
    L2Cache[rowL2].put(loc, memory, L2rows)
    print_log_entry("L2", "SW", pc, loc, rowL2)
    return

def main():
    parser = argparse.ArgumentParser(description='Simulate E20 cache')
    parser.add_argument('filename', help=
        'The file containing machine code, typically with .bin suffix')
    parser.add_argument('--cache', help=
        'Cache configuration: size,associativity,blocksize (for one cache) '
        'or size,associativity,blocksize,size,associativity,blocksize (for two caches)')
    cmdline = parser.parse_args()

    # intialize if there is cache
    oneCache = False
    twoCache = False

    if cmdline.cache is not None:
        parts = cmdline.cache.split(",")
        if len(parts) == 3:
            [L1size, L1assoc, L1blocksize] = [int(x) for x in parts]
            # Simulate one cache
            oneCache = True
            L1rows = (L1size // (L1blocksize * L1assoc))
            L1Cache = [LRUCache(L1assoc, L1blocksize) for row in range(L1rows)] 
            print_cache_config("L1", L1size, L1assoc, L1blocksize, L1rows)
        elif len(parts) == 6:
            [L1size, L1assoc, L1blocksize, L2size, L2assoc, L2blocksize] = [int(x) for x in parts]
            # Simulate two caches
            twoCache = True
            L1rows = (L1size // (L1blocksize * L1assoc))
            L2rows = (L2size // (L2blocksize * L2assoc))
            L1Cache  = [LRUCache(L1assoc, L1blocksize) for row in range(L1rows)]
            L2Cache  = [LRUCache(L2assoc, L2blocksize) for row in range(L2rows)]
            print_cache_config("L1", L1size, L1assoc, L1blocksize, L1rows)
            print_cache_config("L2", L2size, L2assoc, L2blocksize, L2rows)
        else:
            raise Exception("Invalid cache config")

    # Run E20 Simulator
    with open(cmdline.filename) as file:
    # Load file and parse using load_machine_code
        memory = [0000000000000000] * constants.MEM_SIZE
        load_machine_code(file, memory)
        
    # intialize program counter and registers
    pc = 0
    regs = [0] * constants.NUM_REGS 

    # E20 Simulation
    while True:
        # Get OpCode
        addr = memory[pc % constants.MEM_SIZE]
        opCode = getOpCode(addr)
        # For instructions add, sub, or, and, slt, jr, nop
        if opCode == 0:
            regA = getRegALocation(addr)
            regB = getRegBLocation(addr)
            regC = getRegCLocation(addr)
            # Get last four bit to determine operation
            lastFour = getLastFourBits(addr)
            # Skip if you try to edit $0
            if regC == 0 and (lastFour == 0 or lastFour == 1 or lastFour == 2 or lastFour == 3 or lastFour == 4):
                pc = incPC(pc)
                continue
            # add opCode
            if lastFour == 0:
                regs[regC] = makeUnsigned(regs[regA] + regs[regB])
                pc = incPC(pc)
            # sub opCode
            elif lastFour == 1:
                regs[regC] = makeUnsigned(regs[regA] - regs[regB])
                pc = incPC(pc)
            # or opCode
            elif lastFour == 2:
                regs[regC] = makeUnsigned(regs[regA] | regs[regB])
                pc = incPC(pc)
            # and opCode
            elif lastFour == 3:
                regs[regC] = makeUnsigned(regs[regA] & regs[regB])
                pc = incPC(pc)
            # slt opCode
            elif lastFour == 4:
                regs[regC] = 1 if regs[regA] < regs[regB] else 0
                pc = incPC(pc)
            # jr opCode
            else:
                pc = getLastThirteenBits(regs[regA])
        # For instruction addi, movi
        elif opCode == 1:
            regSrc = getRegALocation(addr)
            regDst = getRegBLocation(addr)
            imm = getLastSevenBits(addr)
            # Skip if you try to edit $0
            if regDst == 0:
                pc = incPC(pc)
                continue
            regs[regDst] = makeUnsigned(regs[regSrc] + imm)
            pc = incPC(pc)
        # For instructions j, halt
        elif opCode == 2:
            prevPC = pc
            pc = fixPC(getLastThirteenBits(addr))
            # halt program
            if prevPC == pc:
                break
        # For instructions jal
        elif opCode == 3:
            regs[7] = pc + 1
            pc = fixPC(getLastThirteenBits(addr))
        # For instructions lw
        elif opCode == 4:
            if oneCache:
                lwL1(memory, regs, addr, L1Cache, L1blocksize, L1rows, pc)
            elif twoCache:
                lwL1L2(memory, regs, addr, L1Cache, L1blocksize, L1rows, L2Cache, L2blocksize, L2rows, pc)
            else:
                lw(memory, regs, addr)
            pc = incPC(pc)
        # For instructions sw
        elif opCode == 5:
            if oneCache:
                swL1(memory, regs, addr, L1Cache, L1blocksize, L1rows, pc)
            elif twoCache:
                swL1L2(memory, regs, addr, L1Cache, L1blocksize, L1rows, L2Cache, L2blocksize, L2rows, pc)
            else:
                sw(memory, regs, addr)
            pc = incPC(pc)
        # For instructions jeq
        elif opCode == 6:
            regA = getRegALocation(addr)
            regB = getRegBLocation(addr)
            if regs[regA] == regs[regB]:
                pc = fixPC(getLastSevenBits(addr) + pc + 1)
            else:
                pc = incPC(pc)
        # opCode == 7, for instructions slti
        else:
            regSrc = getRegALocation(addr)
            regDst = getRegBLocation(addr)
            imm = signExtend(getLastSevenBitsUnsigned(addr))
            if regDst == 0:
                pc = incPC(pc)
                continue
            regs[regDst] = 1 if regs[regSrc] < imm else 0
            pc = incPC(pc)
    # Print the final state of the simulator before ending
    # print_state(pc, regs, memory, 128)


if __name__ == "__main__":
    main()
#ra0Eequ6ucie6Jei0koh6phishohm9
