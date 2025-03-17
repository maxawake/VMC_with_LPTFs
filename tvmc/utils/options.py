import json


# # Estimating the Rydberg Hamiltonian:
class Options:
    """Base class for managing options"""

    def __init__(self, **kwargs):
        self.__dict__.update(self.get_defaults())
        self.__dict__.update(kwargs)

    def get_defaults(self):
        """This is where you define your default parameters"""
        return dict()

    def __str__(self):
        out = ""
        for key in self.__dict__:
            line = key + " " * (30 - len(key)) + "\t" * 3 + str(self.__dict__[key])
            out += line + "\n"
        return out

    def cmd(self):
        """Returns a string with command line arguments corresponding to the options
        Outputs:
            out (str) - a single string of space-separated command line arguments
        """
        out = ""
        for key in self.__dict__:
            line = key + "=" + str(self.__dict__[key])
            out += line + " "
        return out[:-1]

    def apply(self, args, warn=True):
        """Takes in a tuple of command line arguments and turns them into options
        Inputs:
            args (tuple<str>) - Your command line arguments

        """
        kwargs = dict()
        for arg in args:
            try:
                key, val = arg.split("=")
                kwargs[key] = self.cmd_cast(val)
                if warn and (not key in self.__dict__):
                    print("Unknown Argument: %s" % key)
            except:
                pass
        self.__dict__.update(kwargs)

    def cmd_cast(self, x0):
        """Casting from a string to other datatypes
        Inputs
            x0 (string) - A string which could represent an int or float or boolean value
        Outputs
            x (?) - The best-fitting cast for x0
        """
        try:
            if x0 == "True":
                return True
            elif x0 == "False":
                return False
            elif x0 == "None":
                return None
            x = x0
            x = float(x0)
            x = int(x0)
        except:
            return x
        return x

    def from_file(self, fn):
        """Depricated: Takes files formatted in the __str__ format and turns them into a set of options
        Instead of using this, consider using save() and load() functions.
        """
        kwargs = dict()
        with open(fn, "r") as f:
            for line in f:
                line = line.strip()
                split = line.split("\t")
                key, val = split[0].strip(), split[-1].strip()
                try:
                    kwargs[key] = self.cmd_cast(val)
                except:
                    pass
        self.__dict__.update(kwargs)

    def save(self, fn):
        """Saves the options in json format
        Inputs:
            fn (str) - The file destination for your output file (.json is not appended automatically)
        Outputs:
            A plain text json file
        """
        with open(fn, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def load(self, fn):
        """Saves  options stored in json format
        Inputs:
            fn (str) - The file source (.json is not appended automatically)
        """
        with open(fn, "r") as f:
            kwargs = json.load(f)
        self.__dict__.update(kwargs)

    def copy(self):
        return Options(**self.__dict__)


class OptionManager:
    registry = dict()

    @staticmethod
    def register(name: str, opt: Options):
        OptionManager.registry[name.upper()] = opt

    @staticmethod
    def parse_cmd(args: list) -> dict:
        output = dict()
        sub_args = []
        for arg in args[::-1]:
            # --name Signifies a new set of options
            if arg[:2] == "--":
                arg = arg.upper()
                # make sure the name is registered
                # if arg[2:] not in OptionManager.registry:
                #     raise Exception("Argument %s Not Registered" % arg)
                # copy the defaults and apply the new options
                opt = OptionManager.registry[arg[2:]].copy()
                opt.apply(sub_args)
                output[arg[2:]] = opt
                # Reset the collection of arguments
                sub_args = []
            # otherwise keep adding options
            else:
                sub_args += [arg]
        return output

class TrainOpt(Options):
    """
    Training Arguments:

        L          (int)     -- Total lattice size (8x8 would be L=64).

        Q          (int)     -- Number of minibatches per batch.

        K          (int)     -- size of each minibatch.

        B          (int)     -- Total batch size (should be Q*K).

        NLOOPS     (int)     -- Number of loops within the off_diag_labels function. Higher values save ram and
                                generally makes the code run faster (up to 2x). Note, you can only set this
                                as high as your effective sequence length. (Take L and divide by your patch size).

        steps      (int)     -- Number of training steps.

        dir        (str)     -- Output directory, set to <NONE> for no output.

        lr         (float)   -- Learning rate.

        seed       (int)     -- Random seed for the run.

        sgrad      (bool)    -- Whether or not to sample with gradients, otherwise create gradients in extra network run.
                                (Uses less ram when but slightly slower)

        true_grad  (bool)    -- Set to false to approximate the gradients, more efficient but approximate.

        sub_directory (str)  -- String to add to the end of the output directory (inside a subfolder).

    """

    def get_defaults(self):
        return dict(
            L=16,
            Q=1,
            K=256,
            B=256,
            NLOOPS=1,
            steps=50000,
            dir="out",
            lr=5e-4,
            seed=None,
            sgrad=False,
            true_grad=False,
            sub_directory="",
        )
