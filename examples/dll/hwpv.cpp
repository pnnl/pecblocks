#include <iostream>
//using namespace std;
#include <fstream>
#include "json.hpp"
using json = nlohmann::json;

int main ()
{
    std::ifstream f("c:\\src\\pecblocks\\examples\\pi\\balanced_fhf.json");
    json data = json::parse(f);
    std::cout << data.dump(4) << std::endl;
}
