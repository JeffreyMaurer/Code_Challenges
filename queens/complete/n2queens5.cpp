#include<iostream>
#include<vector>
#include<list>
#include<algorithm>
#include<fstream>
#include<string>
#include<bitset>
#include<utility>
#include<cmath>
#include<chrono>
#include<climits>
using namespace std;

const int UP    = 0x01;
const int DOWN  = 0x02;
const int RIGHT = 0x04;
const int LEFT  = 0x08;

using Stack = vector<vector<int> >; //list

Stack get_boards(int n, string& fname);
void backtrack(const Stack& boards, const Stack& stack);
bool is_good(const Stack& stack, const vector<int>& board);
inline void roll(vector<int>& in, int where);
void print_solution(const Stack& stack);

ostream& operator<<(ostream& os, const vector<int>& out)  
{  
    for (auto& r : out)
    {
        for (int c = 0; c < out.size()-1; c++)
        {
            if (r==c)  os << "1 "; //add
            else       os << "0 ";
        }
        if (r==(out.size()-1))  os << "1"; //add
        else                    os << "0";
        os << "\n";
    }
    os << "\n";
    return os;  
}

bool operator|(const vector<int>& left, const vector<int>& right) //replace
{
    for (int i = 0; i < left.size(); i++)
    {
        if (left[i]==right[i]) return true; //add
    }
    return false;
}

int main (int argc, char** argv)
{
    int n = atoi(argv[1]);
    string fname = argv[2];
    auto boards = get_boards(n, fname);
    if (boards.size() == 0) { cerr << "mal" << endl; return 1; }
    auto t1 = chrono::high_resolution_clock::now();
    Stack stack {boards[1]};
    boards.erase(boards.begin());
    backtrack(boards, stack);
    auto t2 = chrono::high_resolution_clock::now();
    auto time_span = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    std::cerr << "It took me " << time_span.count() << " seconds." << endl;
    cerr << "Done!!" << endl;
    return 0;
}

Stack get_boards(int n, string& fname)
{
    cerr << "getting subset" << endl;
    int slope = stoi(fname.substr(fname.find(".")+1));
    string rname = to_string(n) + "/" + to_string(n) + "." + to_string(slope) + ".1.R.subset";
    ifstream r((rname).c_str());
    if (r.good()) cout << "rotate" << endl;
    else          cout << "norote" << endl;
    ifstream sub((fname).c_str());
    Stack set; //list of vectors
    string line = "";
    while (getline(sub,line))
    {
        vector<int>  board;
        while (line!="")
        {
            int where = line.find("1") / 2;
            board.push_back(where); //add
            getline(sub,line);
        }
        if (board.size()) set.push_back(board); //add
    }
    return set;
}

void backtrack(const Stack& boards, const Stack& stack)
{
    cerr << stack.size() << endl;
    for (int i = 0; i < boards.size(); i++)
    {
        if (is_good(stack, boards[i])) //insert front
        {
            auto new_stack = stack;
            new_stack.insert(new_stack.begin(),boards[i]); //push_front list
            if (new_stack.size() == boards[i].size()) 
            {
                print_solution(new_stack);
                return; 
            }
            auto new_boards = boards;
            new_boards.erase(new_boards.begin() + i); //remove list
            backtrack(new_boards, new_stack);
        }
    }
}
//instead of 8*layer, it's just 8 now
bool is_good(const Stack& stack, const vector<int>& board)
{
    auto dirs = {UP, DOWN, LEFT, RIGHT, UP|LEFT, UP|RIGHT, DOWN|LEFT, DOWN|RIGHT};
    for (auto& d : dirs)
    {
        auto tmp_board = board;
        for (auto& layer : stack)
        {
            roll(tmp_board, d);
            if (tmp_board|layer) return false;
        }
    }
    return true;
}

inline void roll(vector<int>& in, int where)
{
    if      (where&UP)   
    { 
        in.erase(in.begin(), in.begin()+1); 
        in.push_back(-INT_MAX); //add
    }
    else if (where&DOWN) 
    { 
        in.resize(in.size()-1); 
        in.insert(in.begin(),-INT_MAX); //add 
    }
    for (auto& i : in)
    {
        if      (where&RIGHT) i += 1; //add
        else if (where&LEFT)  i -= 1; //add
    }
}

void print_solution(const Stack& stack)
{
    auto n = stack.size();
    vector<int> board(n*n);
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            for (int layer = 0; layer < n; layer++) //list
            {
                if (stack[layer][row] == col) { board[row*n+col] = layer; break; } //add
            }
        }
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n-1; j++)
        {
            cout << board[i*n+j] << " ";
        }
        cout << board[i*n+n-1] << endl;
    }
    cout << endl;
}
