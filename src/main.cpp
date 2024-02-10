#include "evolutionary.h"

int main() {
    Evolutionary evo("../config/config.yaml");
    evo.evolve("../config/config.yaml");
    return 0;
}
