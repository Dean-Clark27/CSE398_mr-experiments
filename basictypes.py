from copy import deepcopy
from copy import copy


# Variable representations; can take any value
class Variable:
    def __init__(self, name) -> None:
        self.name = name
        pass

    def __str__(self) -> str:
        return f"{self.name}"

    def __repr__(self):
        return str(self)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Variable):
            return False
        return self.name == __o.name

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __deepcopy__(self, memodict = {}):
        return Variable(self.name)

    def __copy__(self):
        return Variable(self.name)


# Predicate representations;  takes 1+ arguments and evaluates to true or false
class Predicate:
    def __init__(self, arity, name) -> None:
        self.name = name
        self.arity = arity

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Predicate):
            return NotImplementedError
        return self.name == __o.name and self.arity == __o.arity

    def __str__(self) -> str:
        return self.name

    def __repr__(self):
        return str(self)

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __deepcopy__(self, memodict={}):
        return Predicate(self.arity, self.name)

    def __copy__(self):
        return Predicate(self.arity, self.name)

    def get_pred_arity_str(self):
        return self.name + "/" + str(self.arity)


# Constant representations; fixed value
class Constant:
    def __init__(self, name) -> None:
        self.name = name

    def __str__(self) -> str:
        return f"{self.name}"

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Constant):
            return False
        return self.name == __o.name

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __deepcopy__(self, memodict={}):
        return Constant(self.name)

    def __copy__(self):
        return Constant(self.name)

    def __repr__(self):
        return str(self)


# Atom representations; consists of a predicate and its arguments
class Atom:
    def __init__(self, pred: Predicate, args: list) -> None:
        self.predicate = pred
        self.arguments = args
        self.arity = self.predicate.arity
    
    def __str__(self) -> str:
        ret = str(self.predicate) + "("
        if self.arguments:
            for i in range(len(self.arguments)):
                ret += str(self.arguments[i]) 
                if i < len(self.arguments)-1:
                    ret += ", "
        return ret + ")"
    
    def get_vars(self):
        """
        Returns the names of all variables in the atom
        :return:
        """
        names = []
        for item in self.arguments:
            if isinstance(item,Variable) and item.name not in names:
                names.append(item.name)
        return names
    
    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Atom):
            return False
        if self.predicate == __o.predicate:
            for i in range(len(self.arguments)):
                if self.arguments[i] != __o.arguments[i]:
                    return False
            return True
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __deepcopy__(self, memodict={}):
        return Atom(deepcopy(self.predicate),
                    [deepcopy(x) for x in self.arguments])

    def __copy__(self):
        return Atom(copy(self.predicate),
                    [copy(x) for x in self.arguments])

    def __repr__(self):
        return str(self)

    def get_pred_arity_string(self):
        return self.predicate.get_pred_arity_str()

    def is_ground(self) -> bool:
        """
        Returns true if the atom has no variables.
        :return:
        """
        ground = True
        for item in self.arguments:
            if isinstance(item,Variable):
                ground = False
                break
        return ground

    def dosubst(self, subst: dict):
        """
        Returns a copy of atom object with subst dictionary applied
        :param subst: a dictionary that maps vars in the atom to new terms
        :return: a new Atom
        """
        return Atom(
            self.predicate,
            [subst.get(arg, arg) for arg in self.arguments]
        )
