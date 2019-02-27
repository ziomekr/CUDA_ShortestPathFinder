#pragma once
class FibonacciHeap {

public:
	struct Node {

		bool cutMark;
		int degree, key, value;
		Node* parent;
		Node* child;
		Node* sibling_left;
		Node* sibling_right;
		Node();
		Node(int _key, int _value);
		void insertSibling(Node* toAdd);
		void removeFromSiblings();
	};

	Node* minimum;
	Node* root_list;
	int size;

	FibonacciHeap();
	void Insert(Node* _toInsert);
	void removeFromRootList(Node* _toRemove);
	Node* extractMinimum();
	void Consolidate();
	void Link(Node* _newChild, Node* _newParent);
	void Cut(Node* _toCut, Node* _from);
	void cascadingCut(Node* _n);
	void decreaseValue(Node* _toDecrease, int _newValue);
};