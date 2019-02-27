//Fibonacci Heap implementation for best performance

#include "FibonacciHeap.h"
#include <ctgmath>
#include <vector>
#include <iostream>


FibonacciHeap::Node::Node()
{
	degree = 0;
	parent = nullptr;
	child = nullptr;
	cutMark = false;
	sibling_left = sibling_right = this;
}

FibonacciHeap::Node::Node(int _key, int _value) : Node()
{
	key = _key;
	value = _value;
}

void FibonacciHeap::Node::insertSibling(Node * _toInsert)
{
	Node* temp = this->sibling_left;
	this->sibling_left = _toInsert;
	_toInsert->sibling_right = this;
	_toInsert->sibling_left = temp;
	temp->sibling_right = _toInsert;

}

void FibonacciHeap::Node::removeFromSiblings()
{
	if (this->sibling_left == this)
		return;
	else {
		
		this->sibling_left->sibling_right = this->sibling_right;
		this->sibling_right->sibling_left = this->sibling_left;
	}
	this->sibling_left = this->sibling_right = this;
}

FibonacciHeap::FibonacciHeap()
{
	this->minimum = this->root_list = nullptr;
	this->size = 0;
}

void FibonacciHeap::Insert(Node * _toInsert)
{
	if (minimum == nullptr) {
		minimum = root_list = _toInsert;
	}
	else {
		root_list->insertSibling(_toInsert);
		if (minimum->value > _toInsert->value)
			minimum = _toInsert;
	}
	this->size += 1;
}

void FibonacciHeap::removeFromRootList(Node * _toRemove)
{
	if (this->root_list == nullptr)
		return;
	if (this->root_list == _toRemove) {
		this->root_list = _toRemove->sibling_left;
		_toRemove->removeFromSiblings();
	}
	else {
		_toRemove->removeFromSiblings();
	}
}

FibonacciHeap::Node* FibonacciHeap::extractMinimum()
{
	Node* toExtract = this->minimum;
	if (toExtract != nullptr) {
		Node* child = toExtract->child;
		Node* nextChild = child;
		if (child != nullptr) {
			do {
				child = nextChild;
				nextChild = child->sibling_left;
				child->removeFromSiblings();
				toExtract->insertSibling(child);
				child->parent = nullptr;
			} while (child != nextChild);
		}
		if (toExtract == toExtract->sibling_left) {
			this->minimum = nullptr;
		}
		else {
			this->removeFromRootList(toExtract);
			this->Consolidate();
		}
		this->size -= 1;
	}
	return toExtract;
}

void FibonacciHeap::Consolidate()
{
	int degree_bound = (int)((log(this->size) / log(0.5*(1 + sqrt(5))))+1);
	std::vector<Node*> aux_vector(degree_bound, nullptr);
	Node* n = this->root_list;
	do{
		int d = n->degree;
		while (aux_vector[d] != nullptr && aux_vector[d]!=n) {
			Node* same_deg_as_n = aux_vector[d];
			if (n->value > same_deg_as_n->value) {
				Node* temp = n;
				n = same_deg_as_n;
				same_deg_as_n = temp;
			}
			this->Link(same_deg_as_n, n);
			aux_vector[d] = nullptr;
			d += 1;
		}
		aux_vector[d] = n;
		n = n->sibling_left;
	} while (n!=root_list);

	this->minimum = nullptr;
	for (int i = 0; i < degree_bound; i++) {
		if (aux_vector[i] != nullptr) {
			if (this->minimum == nullptr) {
				this->root_list = aux_vector[i];
				this->root_list->sibling_left = this->root_list->sibling_right = this->root_list;
				this->minimum = this->root_list;
			}
			else {
				this->root_list->insertSibling(aux_vector[i]);
				if (aux_vector[i]->value < this->minimum->value) {
					this->minimum = aux_vector[i];
				}
			}
		}
	}
}

void FibonacciHeap::Link(Node * _newChild, Node * _newParent)
{
	this->removeFromRootList(_newChild);
	if (_newParent->child == nullptr) {
		_newParent->child = _newChild;
	}
	else {
		_newParent->child->insertSibling(_newChild);
	}
	_newChild->parent = _newParent;
	_newParent->degree += 1;
	_newChild->cutMark = false;
}

void FibonacciHeap::Cut(Node * _toCut, Node * _from)
{
	if (_toCut->sibling_left == _toCut)
		_from->child = nullptr;
	else {
		_from->child = _toCut->sibling_left;
		_toCut->removeFromSiblings();
	}
	_from->degree -= 1;
	this->root_list->insertSibling(_toCut);
	_toCut->parent = nullptr;
	_toCut->cutMark = false;
}

void FibonacciHeap::cascadingCut(Node * n)
{
	Node* parent_of_n = n->parent;
	if (parent_of_n != nullptr) {
		if (!n->cutMark)
			n->cutMark = true;
		else {
			this->Cut(n, parent_of_n);
			cascadingCut(parent_of_n);
		}
	}
}

void FibonacciHeap::decreaseValue(Node * _toDecrease, int _newValue)
{
	if (_newValue > _toDecrease->value)
		return;
	_toDecrease->value = _newValue;
	Node* parent = _toDecrease->parent;
	if (parent != nullptr && _toDecrease->value < parent->value) {
		this->Cut(_toDecrease, parent);
		this->cascadingCut(parent);
	}
	if (_toDecrease->value < this->minimum->value)
		this->minimum = _toDecrease;
}

