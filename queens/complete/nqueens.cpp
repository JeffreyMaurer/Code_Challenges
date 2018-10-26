#include<iostream>
#include<vector>
#include<list>
#include<algorithm>
using namespace std;

typedef vector<int> Board;
typedef vector<int> Col;

void backtrack(Board& board, Col& cols, int row=0);
bool is_good(Board& board, int col, int cRow);
void print_solution(Board& board);
ostream& operator<<(ostream& os, const vector<int>& out)  
{  
    for (auto& r : out)
    {
        for (int c = 0; c < out.size()-1; c++)
        {
            if ((r>>c)&1)  os << "1 ";
            else           os << "0 ";
        }
        if ((r>>(out.size()-1))&1)  os << "1";
        else                        os << "0";
        os << "\n";
    }
    os << "\n";
    return os;  
}

inline int lshift(int i, int j)
{
    if (j>=0) return (i<<j);
    else      return (i>>-j);
}

inline int rshift(int i, int j)
{
    if (j>=0) return (i>>j);
    else      return (i<<-j);
}

int main(int argc, char** argv) 
{
    int n = atoi(argv[1]);
    Board board(n);
    Col cols;
    for (int i = 0; i < n; i++) { cols.push_back(i); }
    backtrack(board, cols);
    cerr << "Done!!" << endl;
    return 0;
}

bool is_good(Board& board, int col, int cRow)
{
    int diagonalRight = 0;
    int diagonalLeft = 0;
    bool debug = false;
    for (int row = 0; row < cRow; row++)
    {
        diagonalRight = diagonalRight | lshift(board[row],(row-cRow));
        diagonalLeft = diagonalLeft | rshift(board[row],(row-cRow));
    }
    return !((1<<col) & (diagonalRight | diagonalLeft));
}

void backtrack(Board& board, Col& cols, int row)
{
    for (auto& col : cols)
    {
        if (is_good(board, col, row))
        {
            Board new_board = board;
            new_board[row] = 1<<col;
            if ((row+1) == board.size())
            {
                cout << new_board;
                return;
            }
            Col new_cols = cols;
            new_cols.erase(remove(new_cols.begin(), new_cols.end(), col), new_cols.end());
            //new_cols.remove(col);
            backtrack(new_board, new_cols, row+1);
        }
    }
}

