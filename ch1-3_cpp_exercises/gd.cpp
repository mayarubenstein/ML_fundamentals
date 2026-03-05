#include <iostream>
#include <iomanip>

double f(double w) {
    double v = w-3;
    return v*v;
}

double grad_f(double w){
    return 2 * (w - 3);
}

int main(){
    double eta;
    int iter;
    std::cin >> eta >> iter;
    double w = 0;
    for (int i = 0; i< iter; i++) {
        w = w - eta * grad_f(w);
        if (i % 100 == 0){
            std::cout <<  "w: ";
            std::cout << std::fixed << std::setprecision(4) << w;
            std::cout << ", f(w): ";
            std::cout << std::fixed << std::setprecision(4) << f(w);
            std::cout << "\n";
        }
    }
    std::cout << "Final w: " << w << "\n";
}