use crate::Priority;
use ecow::EcoVec;
use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct RegisterTree {
    storage: RegisterTreeStorage,
    max_priority: Priority,
}

#[derive(Clone, Debug)]
pub struct RegisterTreeStorage {
    nodes: Vec<RegisterTreeNode>
}

impl RegisterTree {
    pub fn new(max_priority: Priority) -> Self {
        Self {
            storage: RegisterTreeStorage::new(max_priority),
            max_priority,
        }
    }

    /// 2021 register reset
    ///
    /// Assumes all possible registers have already been cached.
    pub fn next_registers(
        &self,
        current: &[Priority],
        vertex_priority: Priority,
        n_registers: usize,
        reset_register: usize,
    ) -> Option<EcoVec<Priority>> {
        let mut current_node = if reset_register == 0 {
            self.storage.get_node_or_leaf(vertex_priority)
        } else {
            self.storage.get_node_or_leaf(0)
        }?;

        for i in 1..n_registers {
            let node = current_node.try_get_node()?;
            match i.cmp(&reset_register) {
                Ordering::Less => current_node = node.get_node_or_leaf(0)?,
                Ordering::Equal => current_node = node.get_node_or_leaf(vertex_priority)?,
                Ordering::Greater => current_node = node.get_node_or_leaf(current[i].max(vertex_priority))?,
            }
        }

        current_node.get_leaf()
    }

    /// 2021 register reset
    ///
    /// Will create a fresh set of registers
    pub fn next_registers_fresh(
        &mut self,
        current: &[Priority],
        vertex_priority: Priority,
        n_registers: usize,
        reset_register: usize,
    ) -> Option<EcoVec<Priority>> {
        let max_priority = self.max_priority;
        let mut output = ecow::EcoVec::from(current);
        let current_output = output.make_mut();
        let mut current_node = if reset_register == 0 {
            current_output[0] = vertex_priority;
            self.get_node_or_leaf_mut(vertex_priority)
        } else {
            current_output[0] = 0;
            self.get_node_or_leaf_mut(0)
        }?;

        for i in 1..n_registers {
            let node = current_node.get_node(max_priority)?;
            match i.cmp(&reset_register) {
                Ordering::Less => {
                    current_output[i] = 0;
                    current_node = node.get_node_or_leaf_mut(0)?
                },
                Ordering::Equal => {
                    current_output[i] = vertex_priority;
                    current_node = node.get_node_or_leaf_mut(vertex_priority)?
                },
                Ordering::Greater => {
                    current_output[i] = current[i].max(vertex_priority);
                    current_node = node.get_node_or_leaf_mut(current_output[i])?
                },
            };
        }
        
        let real_output = current_node.set_leaf(output.clone());
        // tracing::debug!(?current, ?output, ?real_output, ?reset_register, ?vertex_priority);

        // current_node.set_leaf(output)
        real_output
    }

    pub fn find_leaf(&self, mut path: impl Iterator<Item = Priority>) -> Option<EcoVec<Priority>> {
        let mut current_node = self.storage.get_node_or_leaf(path.next()?)?;
        for priority in path {
            current_node = current_node.try_get_node()?.get_node_or_leaf(priority)?;
        }

        current_node.get_leaf()
    }

    pub fn get_node_or_leaf_mut(&mut self, priority: Priority) -> Option<&mut RegisterTreeNode> {
        self.storage.get_node_or_leaf_mut(priority)
    }

    pub fn insert_leaf(&mut self, leaf: EcoVec<Priority>) -> Option<EcoVec<Priority>> {
        let mut current_node = self.storage.get_node_or_leaf_mut(*leaf.first()?)?;
        for &priority in &leaf[1..] {
            current_node = current_node.get_node(self.max_priority)?.get_node_or_leaf_mut(priority)?;
        }

        current_node.set_leaf(leaf)
    }
}

impl RegisterTreeStorage {
    pub fn new(max_priority: Priority) -> Self {
        Self {
            nodes: vec![RegisterTreeNode::Node(None); max_priority as usize + 1],
        }
    }

    pub fn get_node_or_leaf(&self, priority: Priority) -> Option<&RegisterTreeNode> {
        self.nodes.get(priority as usize)
    }

    pub fn get_node_or_leaf_mut(&mut self, priority: Priority) -> Option<&mut RegisterTreeNode> {
        self.nodes.get_mut(priority as usize)
    }

    /// Insert a leaf at the current node, unless it already exists.
    /// In that case a clone is returned of the existing leaf.
    pub fn insert_leaf_at(&mut self, index: usize, leaf: EcoVec<Priority>) -> Option<EcoVec<Priority>> {
        let current_node = self.nodes.get_mut(index)?;

        current_node.set_leaf(leaf)
    }
}

#[derive(Clone, Debug)]
pub enum RegisterTreeNode {
    Node(Option<RegisterTreeStorage>),
    Leaf(EcoVec<Priority>)
}

impl RegisterTreeNode {
    pub fn get_node(&mut self, max_priority: Priority) -> Option<&mut RegisterTreeStorage> {
        self.initialise_node(max_priority)
    }

    pub fn try_get_node(&self) -> Option<&RegisterTreeStorage> {
        match self {
            RegisterTreeNode::Node(content) => content.as_ref(),
            RegisterTreeNode::Leaf(_) => None
        }
    }

    pub fn get_leaf(&self) -> Option<EcoVec<Priority>> {
        match self {
            RegisterTreeNode::Node(_) => None,
            RegisterTreeNode::Leaf(content) => Some(content.clone())
        }
    }

    pub fn set_leaf(&mut self, leaf: EcoVec<Priority>) -> Option<EcoVec<Priority>> {
        match self {
            RegisterTreeNode::Node(_) => {
                *self = RegisterTreeNode::Leaf(leaf.clone());
                Some(leaf)
            }
            RegisterTreeNode::Leaf(existing) => {
                Some(existing.clone())
            }
        }
    }

    pub fn initialise_node(&mut self, max_priority: Priority) -> Option<&mut RegisterTreeStorage> {
        match self {
            RegisterTreeNode::Node(None) => {
                *self = RegisterTreeNode::Node(Some(RegisterTreeStorage::new(max_priority + 1)));
                self.initialise_node(max_priority)
            }
            RegisterTreeNode::Node(storage) => storage.as_mut(),
            RegisterTreeNode::Leaf(_) => None
        }
    }
}