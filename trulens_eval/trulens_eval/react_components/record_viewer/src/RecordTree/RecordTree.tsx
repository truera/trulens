import { useEffect, useState } from 'react';
import { Streamlit } from 'streamlit-component-lib';
import { StackTreeNode } from '../utils/types';
import { getStartAndEndTimesForNode } from '../utils/treeUtils';
import RecordTreeCellRecursive from './RecordTreeCell';
import FolderTreeView from './FolderTreeView';
import { SimpleTreeView } from '@mui/x-tree-view';
import KeyboardArrowDownRounded from '@mui/icons-material/KeyboardArrowDownRounded';
import KeyboardArrowUpRounded from '@mui/icons-material/KeyboardArrowUpRounded';

type RecordTreeProps = {
  root: StackTreeNode;
};

export default function RecordTree({ root }: RecordTreeProps) {
  const [selectedNode, setSelectedNode] = useState<string>();

  useEffect(() => Streamlit.setComponentValue(selectedNode), [selectedNode]);

  const { timeTaken: totalTime, startTime: treeStart } = getStartAndEndTimesForNode(root);

  return (
    <div>
      <SimpleTreeView
        sx={{ p: 1, overflowY: 'auto' }}
        slots={{
          collapseIcon: KeyboardArrowUpRounded,
          expandIcon: KeyboardArrowDownRounded,
        }}
      >
        <RecordTreeCellRecursive
          selectedNode={selectedNode}
          setSelectedNode={setSelectedNode}
          node={root}
          depth={0}
          totalTime={totalTime}
          treeStart={treeStart}
        />
      </SimpleTreeView>
    </div>
  );
}
