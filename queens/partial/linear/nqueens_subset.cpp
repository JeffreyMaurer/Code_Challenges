#include<iostream>
#include<vector>
#include<list>
#include<string>
#include<algorithm>
using namespace std;

struct Board
{
    vector<int> rows;
    vector<string> id;
};

bool is_good(vector<int>& board, int col, int cRow);
void print_solution(vector<int>& board);
ostream& operator<<(ostream& os, const vector<int>& out)  
{  
    for (auto& r : out)
    {
        for (int c = 0; c < out.size()-1; c++)
        {
            if (r==c)  os << "1 ";
            else       os << "0 ";
        }
        if (r==(out.size()-1))  os << "1";
        else                    os << "0";
        os << "\n";
    }
    os << "\n";
    return os;  
}

vector<int>& operator+(int rhs, vector<int>& lhs);
vector<int>& operator-(int rhs, vector<int>& lhs);
vector<int>& operator%(vector<int>& rhs, int lhs);

int main(int argc, char** argv) 
{
    int n = atoi(argv[1]);
    vector<Board> boards;
    for (int slope = 2; slope < n; slope++)
    {
        Board board;
        for (int row = 0; row < n; row++)
        {
            int col = ((row*slope) % n);
            if (is_good(board.rows, col, row))
            {
                board.rows.push_back(col);
            }
            else
            {
                break;
            }
        }
        if (board.size() == n)
        {
            //for (int i = 0; i < n; i++)
            //R
            boards.push_back(board);
            //S
            //rotated R
            //rotated S
        }
    }
    
    for (int i = 0; i < boards.size() - 1; i++)
    {
        for (int j = i+1; j < boards.size();)
        {
            if (boards[i]==boards[j])
            {
                boards.erase(boards.begin()+j);
                continue;
            }
            j++;
        }
        
    }
    cerr << "Done!!" << endl;
    return 0;
}

bool is_good(Board& board, int col, int cRow)
{
    for (int i = 0; i < board.size(); i++)
    {
        if (board[i] == col) return false;
    }
    int diagonalRight = 0;
    int diagonalLeft = 0;
    for (int row = 0; row < cRow; row++)
    {
        diagonalRight = diagonalRight | (board[row]-(row-cRow));
        diagonalLeft = diagonalLeft | (board[row]+(row-cRow));
    }
    return !(diagonalRight | diagonalLeft);
}

